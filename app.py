from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sqlite3
import smtplib
from email.message import EmailMessage


app = Flask(__name__)


# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "final_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    print("Final model loaded successfully.")

except Exception as e:
    model = None
    print("ERROR loading model:", e)


# ============================================================
# DATABASE SETUP
# ============================================================

DB_NAME = os.path.join(
    os.path.dirname(__file__),
    "flood_system.db"
)


def init_db():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            climate REAL,
            water REAL,
            infrastructure REAL,
            human_impact REAL,
            flood_probability REAL,
            risk_level TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# SAVE PREDICTION LOG
# ============================================================

def save_log(
    climate,
    water,
    infrastructure,
    human_impact,
    probability,
    risk_level
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (
            climate,
            water,
            infrastructure,
            human_impact,
            flood_probability,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        climate,
        water,
        infrastructure,
        human_impact,
        probability,
        risk_level
    ))

    conn.commit()
    conn.close()


# ============================================================
# EMAIL ALERT
# ============================================================
def send_email_alert(
            probability,
            climate,
            water,
            infrastructure,
            human_impact,
            receiver_email
 ):

        sender_email = os.environ.get("FLOOD_ALERT_EMAIL")
        sender_password = os.environ.get("FLOOD_ALERT_PASSWORD")

        # Check sender credentials
        if not sender_email or not sender_password:
            print("\n========================================")
            print("EMAIL ALERT")
            print("========================================")
            print("High flood risk detected!")
            print(f"Flood Probability: {probability:.2f}%")
            print(f"Climate & Weather: {climate}")
            print(f"Water & Drainage: {water}")
            print(f"Infrastructure & Preparedness: {infrastructure}")
            print(f"Population & Human Impact: {human_impact}")
            print("Email sender credentials are not configured.")
            print("========================================\n")

            return False

        # Create email
        message = EmailMessage()

        message["Subject"] = "Flood Detection System - High Risk Alert"
        message["From"] = sender_email
        message["To"] = receiver_email

        message.set_content(
            f"""
    FLOOD DETECTION & ALERT SYSTEM

    A high flood risk has been detected.

    Flood Probability: {probability:.2f}%

    Environmental Assessment:

    Climate & Weather: {climate}/10
    Water & Drainage: {water}/10
    Infrastructure & Preparedness: {infrastructure}/10
    Population & Human Impact: {human_impact}/10

    Risk Level: HIGH

    Please take appropriate precautionary measures.
    """
        )

        try:

            with smtplib.SMTP(
                    "smtp.gmail.com",
                    587
            ) as server:

                server.starttls()

                server.login(
                    sender_email,
                    sender_password
                )

                server.send_message(message)

            print("Email alert sent successfully.")

            return True

        except Exception as e:

            print("Email alert failed:", e)

            return False

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# FLOOD PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return jsonify({
            "error": "Machine learning model is not available."
        }), 500


    try:

        data = request.get_json()
        user_email = data.get("email", "").strip()
        if not user_email or "@" not in user_email:
            return jsonify({
                "error": "Please enter a valid email address."
            }), 400


        # ----------------------------------------------------
        # Get the four grouped inputs
        # ----------------------------------------------------

        climate = float(
            data.get("climate", 0)
        )

        water = float(
            data.get("water", 0)
        )

        infrastructure = float(
            data.get("infrastructure", 0)
        )

        human_impact = float(
            data.get("human_impact", 0)
        )


        # ----------------------------------------------------
        # Validate input values
        # ----------------------------------------------------

        inputs = [
            climate,
            water,
            infrastructure,
            human_impact
        ]


        for value in inputs:

            if value < 0 or value > 10:

                return jsonify({
                    "error": "All values must be between 0 and 10."
                }), 400


        # ----------------------------------------------------
        # Prepare model input
        #
        # Order MUST match training:
        #
        # 1. Climate_Weather
        # 2. Water_Drainage
        # 3. Infrastructure_Preparedness
        # 4. Population_Human_Impact
        # ----------------------------------------------------

        model_input = np.array([[
            climate,
            water,
            infrastructure,
            human_impact
        ]])


        # ----------------------------------------------------
        # Predict flood probability
        # ----------------------------------------------------

        prediction = model.predict(model_input)[0]


        # Keep prediction between 0 and 1

        probability = float(
            np.clip(prediction, 0, 1)
        )


        # Convert to percentage

        probability_percent = round(
            probability * 100,
            2
        )


        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------

        if probability < 0.40:

            risk_level = "Low Risk"

        elif probability < 0.60:

            risk_level = "Moderate Risk"

        else:

            risk_level = "High Risk"


        # ----------------------------------------------------
        # Send alert when risk is high
        # ----------------------------------------------------

        email_sent = False


        if risk_level == "High Risk":

            email_sent = send_email_alert(
                probability_percent,
                climate,
                water,
                infrastructure,
                human_impact,
                user_email
            )


        # ----------------------------------------------------
        # Save prediction to database
        # ----------------------------------------------------

        save_log(
            climate,
            water,
            infrastructure,
            human_impact,
            probability,
            risk_level
        )


        # ----------------------------------------------------
        # Response sent to webpage
        # ----------------------------------------------------

        if risk_level == "High Risk":

            message = (
                "High flood risk detected. "
                "Please take appropriate precautionary measures."
            )

        elif risk_level == "Moderate Risk":

            message = (
                "Moderate flood risk detected. "
                "Continue monitoring the conditions."
            )

        else:

            message = (
                "Low flood risk detected. "
                "Current conditions appear relatively safe."
            )


        return jsonify({

            "success": True,

            "probability": probability_percent,

            "risk_level": risk_level,

            "message": message,

            "email_sent": email_sent

        })


    except ValueError:

        return jsonify({
            "error": "Please enter valid numerical values."
        }), 400


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# VIEW PREDICTION LOGS
# ============================================================

@app.route("/logs", methods=["GET"])
def logs():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            timestamp,
            climate,
            water,
            infrastructure,
            human_impact,
            flood_probability,
            risk_level
        FROM logs
        ORDER BY timestamp DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()

    conn.close()


    logs_data = []


    for row in rows:

        logs_data.append({

            "timestamp": row[0],

            "climate": row[1],

            "water": row[2],

            "infrastructure": row[3],

            "human_impact": row[4],

            "flood_probability":
                round(row[5] * 100, 2),

            "risk_level": row[6]

        })


    return jsonify(logs_data)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("FLOOD DETECTION & ALERT SYSTEM")
    print("========================================")
    print("Starting Flask application...")
    print("========================================\n")

    app.run(
        debug=True,
        port=5000
    )

