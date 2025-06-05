from flask import Flask, jsonify, render_template,request,flash,session,redirect,url_for
import sqlite3
import requests
import pandas as pd
import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from flask_bcrypt import Bcrypt
from flask_socketio import SocketIO
import random
import time
from pymongo import MongoClient
import difflib
from bson import ObjectId


app = Flask(__name__)
bcrypt = Bcrypt(app)
socketio = SocketIO(app)
# Blynk Credentials
BLYNK_AUTH_TOKEN = "IuifQYXvoJWJFMnLlhpx7PuJf5HNzbyh"
VIRTUAL_PIN_SPO2 = "V1"
VIRTUAL_PIN_TEMP = "V2"

# Database initialization
DB_NAME = "health_data.db"
app.secret_key = "your_super_secret_key"

client = MongoClient("mongodb://localhost:27017/")
db = client.hafsana
users = db.users


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS health_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        spo2 INTEGER,
                        temperature REAL)''')
    conn.commit()
    conn.close()

init_db()

# Fetch Latest Data from Blynk
# def fetch_blynk_data():
#     try:
#         url_spo2 = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&pin={VIRTUAL_PIN_SPO2}"
#         url_temp = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&pin={VIRTUAL_PIN_TEMP}"
        
#         spo2_response = requests.get(url_spo2).text.strip()
#         temp_celsius = float(requests.get(url_temp).text.strip())
#         temp_fahrenheit = round((temp_celsius * 9/5) + 32, 2)  # Convert to Fahrenheit
        
#         spo2 = int(spo2_response) if spo2_response.isdigit() else None
#         return spo2, temp_fahrenheit
#     except Exception as e:
#         print("Error fetching data:", e)
#         return None, None

def fetch_and_emit_latest_data():
    while True:
        url_spo2 = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&pin={VIRTUAL_PIN_SPO2}"
        url_temp = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&pin={VIRTUAL_PIN_TEMP}"
        
        spo2_response = requests.get(url_spo2).text.strip()
        temp_celsius = float(requests.get(url_temp).text.strip())
        print("spo2 is ",spo2_response)
        print("temp is",temp_celsius)
        # conn = get_db_connection()
        # cursor = conn.cursor()
        # cursor.execute("SELECT spo2, temperature FROM health_data ORDER BY timestamp DESC LIMIT 1")
        # latest_data = cursor.fetchone()
        # conn.close()
        latest_data = {"spo2":spo2_response,"temperature":temp_celsius}

        if latest_data:
            socketio.emit("update_data", {
                "spo2": latest_data["spo2"],
                "temperature": latest_data["temperature"]
            })
        
        time.sleep(5)  

# Store Data in SQLite (Max 4 Entries per Day)
def store_data(spo2, temperature):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    cursor.execute("SELECT COUNT(*) FROM health_data WHERE timestamp LIKE ?", (f"{date}%",))
    count = cursor.fetchone()[0]
    
    if count < 4:
        cursor.execute("INSERT INTO health_data (timestamp, spo2, temperature) VALUES (?, ?, ?)",
                       (timestamp, spo2, temperature))
        conn.commit()
    conn.close()

@app.route("/latest", methods=["GET"])
def get_latest():
    spo2, temp_fahrenheit = fetch_and_emit_latest_data()
    
    if spo2 is None or temp_fahrenheit is None:
        return jsonify({"message": "Failed to fetch data"}), 500

    return jsonify({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spo2": spo2,
        "temperature": temp_fahrenheit
    })


# Fetch Historical Data from SQLite
def get_historical_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT timestamp, spo2, temperature FROM health_data ORDER BY timestamp", conn)
    conn.close()
    return df

def make_stationary(series):
    p_value = adfuller(series)[1]
    if p_value > 0.05:  # If not stationary, difference the series
        return series.diff().dropna()
    return series

# Predict Next Month's Data Using ARIMA
from prophet import Prophet
import pandas as pd

def train_prophet(series, column_name):
    df_prophet = series.reset_index()[["timestamp", column_name]]
    df_prophet.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast["yhat"].iloc[-30:].values  # Get last 30 days

@app.route("/predict", methods=["GET"])
def predict():
    df = get_historical_data()
    
    if df.empty:
        return jsonify({"message": "No data available for prediction"}), 404

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Debug: Print before resampling
    print("Before resampling:\n", df.head())

    # Try without resampling first
    # df = df.resample("6H").mean().dropna()

    # Debug: Print after resampling
    print("After resampling:\n", df.head())

    print("Historical Data Summary:\n", df.describe())

    spo2_forecast = train_prophet(df, "spo2")
    temp_forecast = train_prophet(df, "temperature")

    print("Raw SpO2 forecast:", spo2_forecast[:10])  # Debug
    print("Raw Temp forecast:", temp_forecast[:10])  # Debug

    temp_forecast = np.clip(temp_forecast, 95, 102)  # Keep realistic range

    future_dates = [(datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
    
    predictions = [{
        "date": future_dates[i],
        "predicted_spo2": round(spo2_forecast[i], 2),
        "predicted_temperature": round(temp_forecast[i], 2)
    } for i in range(30)]
    
    print("Final Predictions:", predictions[:5])  # Debug first 5
    
    return jsonify(predictions)


# Routes for Frontend
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")

@app.route("/history", methods=["GET"])
def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, spo2, temperature FROM health_data ORDER BY timestamp ASC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify([])  

    result = [{"timestamp": row[0], "spo2": row[1], "temperature": row[2]} for row in rows]
    return jsonify(result)

def get_db_connection():
    conn = sqlite3.connect("health.db")
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn




@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        patient_id = request.form["patient_id"].strip()
        first_name = request.form["first_name"].strip()
        last_name = request.form["last_name"].strip()
        dob = request.form["dob"].strip()
        gender = request.form["gender"].strip()
        phone = request.form["phone"].strip()
        blood_group = request.form["blood_group"].strip()
        username = request.form["username"].strip()
        usermailid = request.form["usermailid"].strip()
        password = request.form["password"].strip()
        emergency_contact = {
            "full_name": request.form["emergency_name"].strip(),
            "phone": request.form["emergency_phone"].strip(),
            "email": request.form["emergency_email"].strip(),
            "address": request.form["emergency_address"].strip()
        }
        
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
        
        if users.find_one({"username": username}):
            flash("Username already exists!", "danger")
        else:
            users.insert_one({
                "patient_id": patient_id,
                "first_name": first_name,
                "last_name": last_name,
                "dob": dob,
                "gender": gender,
                "phone": phone,
                "blood_group": blood_group,
                "username": username,
                "usermailid": usermailid,
                "password": hashed_pw,
                "emergency_contact": emergency_contact,
                "user_type": 'Normal'
            })
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        user = users.find_one({"username": username})
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user"] = username
            flash("Login successful!", "success")
            return redirect(url_for("about"))
        flash("Invalid credentials!", "danger")
    return render_template("login.html")


@app.route("/about")
def about():

    return render_template("about.html") 

@app.route("/logout")
def logout():
    if "user" in session:
        session.pop("user", None)
        flash("Logged out successfully!", "info")
    else:
        flash("You are not logged in!", "warning")
    
    return redirect(url_for("login"))

@socketio.on("connect")
def handle_connect():
    print("Client connected!")



responses = {
    "hi": "Hello! How can I assist you with your health today?",
    "hello": "Hi there! What health-related query do you have?",
    "how are you": "I'm just a bot, but I'm here to help you with health-related questions!",
    "what is fever": "Fever is a temporary increase in body temperature, often due to an illness.",
    "symptoms of covid": "Common symptoms include fever, cough, shortness of breath, fatigue, and loss of taste or smell.",
    "how to reduce fever": "You can take paracetamol, drink plenty of fluids, and get rest. If symptoms persist, consult a doctor.",
    "healthy diet": "A healthy diet includes fruits, vegetables, whole grains, lean proteins, and plenty of water.",
    "exercise benefits": "Regular exercise improves heart health, boosts immunity, and helps maintain a healthy weight.",
    "mental health tips": "Practice mindfulness, get enough sleep, stay active, and talk to someone if you feel overwhelmed.",
    "headache remedies": "Try drinking water, resting, reducing screen time, or taking a mild pain reliever if needed.",
}

def get_best_match(query):
    query = query.lower()
    best_match = difflib.get_close_matches(query, responses.keys(), n=1, cutoff=0.5)
    print(best_match)
    return responses.get(best_match[0], "I'm not sure about that. Can you try asking in a different way?") if best_match else "I don't have an answer for that yet."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")
    print("user quey is",user_query)
    response = get_best_match(user_query)
    print(response)
    return jsonify({"response": response})


help_requests = db["help_requests"]
@app.route('/request_help', methods=['POST'])
def request_help():
    data = request.json
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    
    if latitude is None or longitude is None:
        return jsonify({"message": "Invalid location data"}), 400

    help_requests.insert_one({
        "user_id": session.get("user", "guest"),  # Replace with actual user ID
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": datetime.datetime.utcnow(),
        "status": "Pending"
    })

    return jsonify({"message": "Help request sent successfully!"})


# @app.route('/ambulance_dashboard')
# def ambulance_dashboard():
#     requests = list(help_requests.find({}))  # Exclude MongoDB's _id field
#     return render_template('ambulance_dashboard.html', requests=requests)


@app.route('/take_request', methods=['POST'])
def take_request():
    data = request.json
    request_id = data.get("request_id")
    
    if not request_id:
        return jsonify({"success": False, "message": "Invalid request ID"}), 400

    help_requests.update_one({"_id": ObjectId(request_id)}, {"$set": {"status": "In Progress"}})
    return jsonify({"success": True, "message": "Request updated!"})


@app.route('/mark_done', methods=['POST'])
def mark_done():
    if "driver_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    data = request.json
    request_id = data.get("request_id")

    request_data = help_requests.find_one({"_id": ObjectId(request_id)})
    if not request_data:
        return jsonify({"success": False, "message": "Request not found"}), 404

    # Mark request as completed
    help_requests.update_one({"_id": ObjectId(request_id)}, {"$set": {"status": "Completed"}})

    # Save in driver's past trips
    drivers.update_one(
        {"_id": ObjectId(session["driver_id"])},
        {"$push": {"past_trips": request_data}}
    )

    return jsonify({"success": True, "message": "Request marked as completed!"})


drivers = db["drivers"]
@app.route('/dregister', methods=['POST','GET'])
def dregister():
    if request.method =='GET':
        return  render_template('dregister.html')
    else:
        data = request.json
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if drivers.find_one({"email": email}):
            return jsonify({"success": False, "message": "Email already registered"}), 400

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        drivers.insert_one({"name": name, "email": email, "password": hashed_password, "past_trips": []})

        return jsonify({"success": True, "message": "Driver registered!"})

@app.route('/dlogin', methods=['POST','GET'])
def dlogin():
    if request.method =='GET':
        return  render_template('dlogin.html')
    else:
        data = request.json
        email = data.get("email")
        password = data.get("password")

        driver = drivers.find_one({"email": email})
        if not driver or not bcrypt.check_password_hash(driver["password"], password):
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        session["driver_id"] = str(driver["_id"])
        return jsonify({"success": True, "message": "Login successful"})

@app.route('/ambulance_dashboard')
def ambulance_dashboard():
    if "driver_id" not in session:
        return redirect('/login')

    requests = list(help_requests.find({}))
    return render_template("ambulance_dashboard.html", requests=requests)


@app.route('/profile')
def profile():
    if "driver_id" not in session:
        return redirect('/login')

    driver = drivers.find_one({"_id": ObjectId(session["driver_id"])})
    return render_template("profile.html", driver=driver)


if __name__ == "__main__":
    socketio.start_background_task(fetch_and_emit_latest_data)
    socketio.run(app, debug=True)
