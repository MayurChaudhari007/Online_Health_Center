from flask import (
    Flask,
    request,
    redirect,
    render_template,
    url_for,
    jsonify,
    send_from_directory,
    flash,
    abort,
    session,
    send_file,
)

# from yourapp import app, db
import io
import concurrent.futures
import logging
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

#
# from google import genai as ge
# import math
import ast

# import openai
import google.generativeai as genai

# import requests
import json

from uuid import uuid4
import os
import re
import importlib

# Load google-genai's genai
from dotenv import load_dotenv

ge = importlib.import_module("google.genai")

# Import the library

# Load environment variables from .env file
load_dotenv()


############### Api Key ###################


################ DATAsets ################

sym_desk = pd.read_csv("Dataset/symtoms_df.csv")
precautions = pd.read_csv("Dataset/precautions_df.csv")
workout = pd.read_csv("Dataset/workout_df.csv")
description = pd.read_csv("Dataset/description.csv")
medications = pd.read_csv("Dataset/medications.csv")
diets = pd.read_csv("Dataset/diets.csv")
# Extra
medications1 = pd.read_csv("Dataset/medications.csv")
medications1["Medication"] = medications1["Medication"].apply(ast.literal_eval)
diets1 = pd.read_csv("Dataset/diets.csv")
diets1["Diet"] = diets1["Diet"].apply(ast.literal_eval)


#  load model
svc = pickle.load(open("model/svc.pkl", "rb"))


# # function
def medi(dis):
    results = []
    if "Disease" in medications1.columns:
        dis_type = medications1[medications1["Disease"] == dis]
        for med_list in dis_type["Medication"]:
            for med in med_list:
                results.append(med)
    return results


def dieti(dis):
    results = []
    if "Disease" in diets1.columns:
        diet_type = diets1[diets1["Disease"] == dis]
        for diet_list in diet_type["Diet"]:
            for die in diet_list:
                results.append(die)
    return results


def helper(dis):
    desc = description[description["Disease"] == dis]["Description"]
    desc = " ".join([w for w in desc])

    pre = precautions[precautions["Disease"] == dis][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ]

    pre = pre.values.flatten().tolist()

    wrkout = workout[workout["disease"] == dis]["workout"]

    return desc, pre, wrkout


symptoms_dict = {
    "itching": 0,
    "skin_rash": 1,
    "nodal_skin_eruptions": 2,
    "continuous_sneezing": 3,
    "shivering": 4,
    "chills": 5,
    "joint_pain": 6,
    "stomach_pain": 7,
    "acidity": 8,
    "ulcers_on_tongue": 9,
    "muscle_wasting": 10,
    "vomiting": 11,
    "burning_micturition": 12,
    "spotting_ urination": 13,
    "fatigue": 14,
    "weight_gain": 15,
    "anxiety": 16,
    "cold_hands_and_feets": 17,
    "mood_swings": 18,
    "weight_loss": 19,
    "restlessness": 20,
    "lethargy": 21,
    "patches_in_throat": 22,
    "irregular_sugar_level": 23,
    "cough": 24,
    "high_fever": 25,
    "sunken_eyes": 26,
    "breathlessness": 27,
    "sweating": 28,
    "dehydration": 29,
    "indigestion": 30,
    "headache": 31,
    "yellowish_skin": 32,
    "dark_urine": 33,
    "nausea": 34,
    "loss_of_appetite": 35,
    "pain_behind_the_eyes": 36,
    "back_pain": 37,
    "constipation": 38,
    "abdominal_pain": 39,
    "diarrhoea": 40,
    "mild_fever": 41,
    "yellow_urine": 42,
    "yellowing_of_eyes": 43,
    "acute_liver_failure": 44,
    "fluid_overload": 45,
    "swelling_of_stomach": 46,
    "swelled_lymph_nodes": 47,
    "malaise": 48,
    "blurred_and_distorted_vision": 49,
    "phlegm": 50,
    "throat_irritation": 51,
    "redness_of_eyes": 52,
    "sinus_pressure": 53,
    "runny_nose": 54,
    "congestion": 55,
    "chest_pain": 56,
    "weakness_in_limbs": 57,
    "fast_heart_rate": 58,
    "pain_during_bowel_movements": 59,
    "pain_in_anal_region": 60,
    "bloody_stool": 61,
    "irritation_in_anus": 62,
    "neck_pain": 63,
    "dizziness": 64,
    "cramps": 65,
    "bruising": 66,
    "obesity": 67,
    "swollen_legs": 68,
    "swollen_blood_vessels": 69,
    "puffy_face_and_eyes": 70,
    "enlarged_thyroid": 71,
    "brittle_nails": 72,
    "swollen_extremeties": 73,
    "excessive_hunger": 74,
    "extra_marital_contacts": 75,
    "drying_and_tingling_lips": 76,
    "slurred_speech": 77,
    "knee_pain": 78,
    "hip_joint_pain": 79,
    "muscle_weakness": 80,
    "stiff_neck": 81,
    "swelling_joints": 82,
    "movement_stiffness": 83,
    "spinning_movements": 84,
    "loss_of_balance": 85,
    "unsteadiness": 86,
    "weakness_of_one_body_side": 87,
    "loss_of_smell": 88,
    "bladder_discomfort": 89,
    "foul_smell_of urine": 90,
    "continuous_feel_of_urine": 91,
    "passage_of_gases": 92,
    "internal_itching": 93,
    "toxic_look_(typhos)": 94,
    "depression": 95,
    "irritability": 96,
    "muscle_pain": 97,
    "altered_sensorium": 98,
    "red_spots_over_body": 99,
    "belly_pain": 100,
    "abnormal_menstruation": 101,
    "dischromic _patches": 102,
    "watering_from_eyes": 103,
    "increased_appetite": 104,
    "polyuria": 105,
    "family_history": 106,
    "mucoid_sputum": 107,
    "rusty_sputum": 108,
    "lack_of_concentration": 109,
    "visual_disturbances": 110,
    "receiving_blood_transfusion": 111,
    "receiving_unsterile_injections": 112,
    "coma": 113,
    "stomach_bleeding": 114,
    "distention_of_abdomen": 115,
    "history_of_alcohol_consumption": 116,
    "fluid_overload.1": 117,
    "blood_in_sputum": 118,
    "prominent_veins_on_calf": 119,
    "palpitations": 120,
    "painful_walking": 121,
    "pus_filled_pimples": 122,
    "blackheads": 123,
    "scurring": 124,
    "skin_peeling": 125,
    "silver_like_dusting": 126,
    "small_dents_in_nails": 127,
    "inflammatory_nails": 128,
    "blister": 129,
    "red_sore_around_nose": 130,
    "yellow_crust_ooze": 131,
}
diseases_list = {
    15: "Fungal infection",
    4: "Allergy",
    16: "GERD",
    9: "Chronic cholestasis",
    14: "Drug Reaction",
    33: "Peptic ulcer diseae",
    1: "AIDS",
    12: "Diabetes ",
    17: "Gastroenteritis",
    6: "Bronchial Asthma",
    23: "Hypertension ",
    30: "Migraine",
    7: "Cervical spondylosis",
    32: "Paralysis (brain hemorrhage)",
    28: "Jaundice",
    29: "Malaria",
    8: "Chicken pox",
    11: "Dengue",
    37: "Typhoid",
    40: "hepatitis A",
    19: "Hepatitis B",
    20: "Hepatitis C",
    21: "Hepatitis D",
    22: "Hepatitis E",
    3: "Alcoholic hepatitis",
    36: "Tuberculosis",
    10: "Common Cold",
    34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)",
    18: "Heart attack",
    39: "Varicose veins",
    26: "Hypothyroidism",
    24: "Hyperthyroidism",
    25: "Hypoglycemia",
    31: "Osteoarthristis",
    5: "Arthritis",
    0: "(vertigo) Paroymsal  Positional Vertigo",
    2: "Acne",
    38: "Urinary tract infection",
    35: "Psoriasis",
    27: "Impetigo",
}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# #############################


# ############ AI Prediction Functions #########################


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Configure the Ai prediction functions
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    system_instruction="""
    You are a professional AI doctor.
The user will give you their name, age, gender, and symptoms.

You must respond in the following JSON format only:
{
  "patient_name": "<name>",
  "predicted_disease": "<disease name>",
  "medical_advice": ["advice 1", "advice 2", "advice 3"],
  "additional_info": ["extra info 1", "extra info 2"],
  "doctor_alerts": ["alert 1", "alert 2"]
}

Rules:
- Do not include any extra text outside the JSON dont add json variable while creating json file.
- Keep each advice short and clear.
- Use layman-friendly language.
- If unsure, say “Possible causes include...” in the disease field.
    
    """,
)


# ############ Chat Bot Functions #########################

client = ge.Client(api_key=os.getenv("GEMINI_API_KEY"))


chat = client.chats.create(
    model="gemini-2.0-flash-lite",
    history=[
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "You are a friendly AI doctor. "
                        "Ask clear, short questions if needed. "
                        "Give simple and easy-to-understand medical advice. "
                        "used less line to give answer like 2 or 3 line only"
                        "Also give medical or health advice if neccessary "
                        "Be polite, helpful, and concise."
                    )
                }
            ],
        }
    ],
)

########################################################

app = Flask(__name__)

# #############################
app.secret_key = "your_secret_key"

# ---------------------------
# SQLAlchemy Config
# ---------------------------


# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


## # Render Database Add
# app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# # # Neon Database add


app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL2")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"sslmode": "require"}
}


# # ########
db = SQLAlchemy(app)
# migrate = Migrate(app, db)
# ---------------------------
# File Upload Config
# ---------------------------
# Files will be stored in: <your project> / instance / uploads / <user_id> / <file>
# app.config["UPLOAD_FOLDER"] = os.path.join(app.instance_path, "uploads")
# app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max per file
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "doc", "docx"}

# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    # flash("❌ File too large! Maximum allowed size is 2 MB.")
    flash("File too large! Maximum allowed size is 2 MB.", "danger")
    return redirect(url_for("dashboard"))


# ---------------------------
# Database Models
# ---------------------------

# # # # # #  new database update # # #


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(350), unique=True, nullable=False)
    password_hash = db.Column(db.String(350), nullable=False)

    # This is perfect! The cascade will handle automatic deletion.
    chats = db.relationship("Chat", backref="user", cascade="all, delete-orphan")
    reports = db.relationship(
        "MedicalReport", backref="user", cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # This is the only link you need to the user.
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # ❌ REMOVED: username = db.Column(db.String(400))
    sender = db.Column(db.String(400))  # "user" / "bot"
    message = db.Column(db.Text)


class MedicalReport(db.Model):
    # This model is already correct, no changes needed.
    id = db.Column(db.Integer, primary_key=True)
    original_name = db.Column(db.String(260), nullable=False)
    mimetype = db.Column(db.String(100))
    size = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)


# # # # ## # #

with app.app_context():
    db.create_all()


# ---------------------------
# Jinja Filters
# ---------------------------
@app.template_filter("filesize")
def filesize(num):
    try:
        num = float(num)
    except Exception:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


########################################################


@app.route("/")
def home1():
    # if "username" in session:
    #     return redirect(url_for("home"))
    return render_template("index.html")



#  New Login Code -------------------------------------------------------------


# --- LOGIN ROUTE (Corrected) ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            # ✅ FIX: Store the user's permanent ID in the session.
            session["user_id"] = user.id
            flash("✅ Login successful!", "success")
            return redirect(url_for("home"))  # Or your desired page, e.g., 'profile'
        else:
            # flash("❌ Invalid username or password.", "danger")
            # return redirect(url_for("login")) # Redirect back to the login page
            return render_template("index.html", error="Invalid username or password.")

    # For a GET request, just show the login page
    return render_template("index.html")


# ------------------------------------------------------------------------

# Register (GET shows form page, POST handles submission)


# ------------ new Register page -----------------------------


# --- REGISTER ROUTE (Corrected) ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if User.query.filter_by(username=username).first():
            # flash("❌ Username already exists!", "danger")
            # return redirect(url_for("register"))
            return render_template("register.html", error="User already exists!")

        if password != confirm_password:
            # flash("❌ Passwords do not match!", "danger")
            # return redirect(url_for("register"))
            return render_template("register.html", error="Passwords do not match!")

        # Create the new user
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        # ✅ FIX: Log the new user in by storing their new ID in the session.
        session["user_id"] = new_user.id
        flash(
            f"✅ Welcome, {new_user.username}! Your account has been created.",
            "success",
        )
        return redirect(url_for("home"))  # Or your desired page

    # For a GET request, just show the registration page
    return render_template("register.html")


# -----------------------------------------------------------


###################
# Profile Function and route
#########################


# # # # # # # # # # # # # Delete account function and route


# # # # # new Profile --------------------------------------------------------------------------


@app.route("/profile", methods=["GET", "POST"])
def profile():
    # ✅ FIX: Check for 'user_id' in session.
    if "user_id" not in session:
        flash("⚠️ Please login first.", "danger")
        return redirect(url_for("login"))

    # ✅ FIX: Get the user by their primary key (ID). This is faster and more reliable.
    user = User.query.get(session["user_id"])
    if not user:
        # If user was deleted but session still exists, clear session
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":
        new_username = request.form.get("username")
        # The rest of your update logic is fine, but we no longer need to update the session!
        if new_username and new_username != user.username:
            if User.query.filter_by(username=new_username).first():
                flash("❌ Username already taken!", "danger")
                return redirect(url_for("profile"))
            user.username = new_username
            # ❌ REMOVED: session["username"] = new_username

        # (Password change logic is unchanged and correct)
        # ...
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")

        if old_password and new_password:
            if not check_password_hash(user.password_hash, old_password):
                flash("❌ Old password is incorrect!", "danger")
                return redirect(url_for("profile"))
            if new_password != confirm_password:
                flash("❌ Passwords do not match!", "danger")
                return redirect(url_for("profile"))
            user.password_hash = generate_password_hash(new_password)

        db.session.commit()
        flash("✅ Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=user, username=user.username)


# # # # # # # # # # # # Delete account function and route
@app.route("/delete_account", methods=["POST"])
def delete_account():
    # ✅ FIX: Use 'user_id' from session.
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])

    if user:
        # ✅ FIX: The cascade="all, delete-orphan" in your User model handles everything!
        # These lines are no longer needed. The ORM does it for you.
        # MedicalReport.query.filter_by(user_id=user.id).delete()
        # Chat.query.filter_by(user_id=user.id).delete()

        db.session.delete(
            user
        )  # This one line will delete the user, their chats, and their reports.
        db.session.commit()

        session.clear()  # Log user out completely
        # flash("🗑️ Your account and all data have been deleted successfully.", "success")
        return redirect(url_for("register"))

    flash("❌ Something went wrong!", "danger")
    return redirect(url_for("profile"))


# # #  ----------------------------------------------------------------------------------------------

# Dashboard (upload + list files)


################################new storage -----------------------------


@app.route("/dashboard")
def dashboard():
    # ✅ FIX: Use 'user_id' from session
    if "user_id" not in session:
        return redirect(url_for("home1"))

    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        flash("Session expired. Please log in again.")
        return redirect(url_for("home1"))

    reports = (
        MedicalReport.query.filter_by(user_id=user.id)
        .order_by(MedicalReport.uploaded_at.desc())
        .all()
    )
    return render_template("dashboard.html", user=user, reports=reports)


# All other report routes (upload, download, delete) should also be changed to get the user via:
# user = User.query.get(session["user_id"])
# The rest of your logic inside those functions is correct because it already uses user.id.

#######################---------------------------



##########################################-------------------------------------------------
###############new report code----------------------


# --- UPLOAD ROUTE (Corrected) ---
@app.route("/upload", methods=["POST"])
def upload():
    # ✅ FIX: Check for 'user_id' in the session.
    if "user_id" not in session:
        flash("⚠️ Please login to upload files.", "danger")
        return redirect(url_for("login"))

    # ... (file checking logic is correct and unchanged) ...
    if "report" not in request.files:
        flash("No file part in the form.", "warning")
        return redirect(url_for("dashboard"))
    file = request.files["report"]
    if file.filename == "":
        flash("No file was selected.", "warning")
        return redirect(url_for("dashboard"))
    if not allowed_file(file.filename):
        flash("File type not allowed.", "danger")
        return redirect(url_for("dashboard"))

    # ✅ FIX: Get the user by their primary key (ID).
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("login"))

    safe_name = secure_filename(file.filename)
    data = file.read()
    size = len(data)

    # This part was already correct because it uses user.id
    report = MedicalReport(
        original_name=safe_name,
        mimetype=file.mimetype,
        size=size,
        user_id=user.id,
        data=data,
    )
    db.session.add(report)
    db.session.commit()

    flash("✅ Report uploaded successfully!", "success")
    return redirect(url_for("dashboard"))


##############################  new report download-----------------------------------


# --- DOWNLOAD ROUTE (Corrected) ---
@app.route("/files/<int:report_id>")
def download(report_id):
    # ✅ FIX: Check for 'user_id' in the session.
    if "user_id" not in session:
        flash("⚠️ Please login to download files.", "danger")
        return redirect(url_for("login"))

    # ✅ FIX: Get the user by their primary key (ID).
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    report = MedicalReport.query.get_or_404(report_id)

    # This authorization check was already correct
    if report.user_id != user.id:
        abort(403)  # Access Forbidden

    return send_file(
        io.BytesIO(report.data),
        as_attachment=True,
        download_name=report.original_name,
        mimetype=report.mimetype,
    )


################################## new delete report code --------------------------


# --- DELETE FILE ROUTE (Corrected) ---
@app.route("/files/<int:report_id>/delete", methods=["POST"])
def delete_file(report_id):
    # ✅ FIX: Check for 'user_id' in the session.
    if "user_id" not in session:
        flash("⚠️ Please login to delete files.", "danger")
        return redirect(url_for("login"))

    # ✅ FIX: Get the user by their primary key (ID).
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    report = MedicalReport.query.get_or_404(report_id)

    # This authorization check was already correct
    if report.user_id != user.id:
        abort(403)

    db.session.delete(report)
    db.session.commit()
    flash("🗑️ Report deleted successfully!", "success")
    return redirect(url_for("dashboard"))


###########################---------------------------------------------------
#  # # # ##  Logout -----------------------------------------


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


#####################    Home      #######################################





@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])

   
    if not user:
        session.clear() 
        flash("Your session has expired. Please log in again.", "warning")
        return redirect(url_for("login"))

    return render_template("home.html", user=user)


########################################################
#####################    predict      #######################################



@app.route("/predict", methods=["GET", "POST"])  # Assuming the route is /predict
def form():
    # ✅ FIX: Check if user is logged in using 'user_id'
    if "user_id" not in session:
        flash("⚠️ Please login to get a prediction.", "danger")
        return redirect(url_for("login"))

    # ✅ FIX: Get the user by their permanent ID
    user = User.query.get(session["user_id"])

    # ✅ FIX: Handle case where user was deleted but session exists
    if not user:
        session.clear()
        flash("Your session has expired. Please log in again.", "warning")
        return redirect(url_for("login"))

    # --- POST request logic (when the form is submitted) ---
    if request.method == "POST":
        name = request.form.get("patient_name")
        age = request.form.get("patient_age")
        gender = request.form.get("patient_gender")
        symptoms = request.form.getlist("symptoms")

        if not symptoms:
            message = "Please select at least one symptom."
            return render_template(
                "predict.html",
                message=message,
                symptoms_list=list(symptoms_dict.keys()),
                user=user,  # Pass the user object even on error
            )

        # Predict disease
        predicted_disease = get_predicted_value(symptoms)
        desc, pre, wrkout = helper(predicted_disease)
        medi2 = medi(predicted_disease)
        diet2 = dieti(predicted_disease)
        report_time = datetime.now().strftime("%d-%m-%Y %H:%M")

        return render_template(
            "predict.html",
            name=name,
            age=age,
            gender=gender,
            symptoms=symptoms,
            predicted_disease=predicted_disease,
            report_time=report_time,
            dis_desc=desc,
            dis_prec=pre,
            dis_medi=medi2,
            dis_diet=diet2,
            dis_wrk=wrkout,
            symptoms_list=list(symptoms_dict.keys()),
            user=user,  # ✅ FIX: Pass the entire user object
        )

    # --- GET request logic (when the page is first loaded) ---
    return render_template(
        "predict.html",
        user=user,  # ✅ FIX: Pass the entire user object
        symptoms_list=list(symptoms_dict.keys()),
    )


########################################################################################
######################     AI   Prediction       #######################################



@app.route("/ai_predict", methods=["GET", "POST"])
def ai_predict():
    # ✅ FIX: Check for 'user_id' in the session to ensure the user is logged in.
    if "user_id" not in session:
        flash("⚠️ Please login to use the AI prediction service.", "danger")
        return redirect(url_for("login"))

    # ✅ FIX: Get the user by their permanent ID.
    user = User.query.get(session["user_id"])

    # ✅ FIX: Handle cases where the session is invalid or the user was deleted.
    if not user:
        session.clear()
        flash("Your session has expired. Please log in again.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        symptoms = request.form.get("symptoms")

        user_prompt = f"Name: {name}, Age: {age}, Gender: {gender}, Symptoms: {symptoms}"

        try:
            # (Your AI prediction logic is good and remains unchanged)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(model.generate_content, user_prompt)
                response = future.result(timeout=10)

            prediction = response.text if response else None
            if not prediction:
                raise ValueError("No response from AI")

            cleaned_text = re.sub(r"^```json|```$", "", prediction.strip()).strip("`").strip()
            
            try:
                prediction_data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                prediction_data = {"raw_text": cleaned_text}
            
            result = prediction_data

        except concurrent.futures.TimeoutError:
            app.logger.error("AI Prediction timeout")
            flash("⚠️ AI service timed out. Please try again later.", "danger")
            return redirect(url_for("ai_predict"))
        except Exception as e:
            app.logger.error(f"AI Prediction error: {e}")
            flash("⚠️ AI service is unavailable. Please try again later.", "danger")
            return redirect(url_for("ai_predict"))

        report_time = datetime.now().strftime("%d-%m-%Y %H:%M")

        return render_template(
            "ai_predict.html",
            user=user,  # ✅ FIX: Pass the entire user object to the template.
            result=result,
            report_time=report_time,
        )

    # This is for the GET request (when the page is first loaded)
    return render_template("ai_predict.html", user=user, prediction=None)

##########################################################################################
######################     Chat Bot       #######################################


#  Chatbot page
# @app.route("/chatbot")
# def chatbot_page():
#     if "username" not in session:
#         return redirect(url_for("login"))

#     user = session["username"]
#     history = Chat.query.filter_by(username=user).all()
#     return render_template("chatbot.html", username=user, history=history)


# # Get response route
# @app.route("/get_response", methods=["POST"])
# def get_response():
#     if "username" not in session:
#         return jsonify({"reply": "⚠️ Please login first."})

#     data = request.get_json()
#     user_message = data.get("message")
#     lang = data.get("lang", "en-US")  # 🌐 get language from frontend

#     # Save user message
#     chat_entry = Chat(username=session["username"], sender="user", message=user_message)
#     db.session.add(chat_entry)
#     db.session.commit()

#     # 🔹 Adjust user message for multilingual reply
#     if lang == "hi-IN":
#         user_message = f"Reply in Hindi: {user_message}"
#     elif lang == "mr-IN":
#         user_message = f"Reply in Marathi: {user_message}"

#     # Generate bot reply
#     response = chat.send_message(user_message)
#     bot_reply = response.text

#     # Save bot reply
#     bot_entry = Chat(username=session["username"], sender="bot", message=bot_reply)
#     db.session.add(bot_entry)
#     db.session.commit()

#     return jsonify({"reply": bot_reply})


# # Clear chat
# @app.route("/clear_chat")
# def clear_chat():
#     user = User.query.filter_by(username=session["username"]).first()
#     if "username" in session:
#         # Chat.query.filter_by(username=session["username"]).delete()
#         Chat.query.filter_by(user_id=user.id).delete()
#         db.session.commit()
#         flash("Chat is cleared. Scroll down to start fresh.", "success")
#     return redirect(url_for("chatbot_page"))

################################## new chat functions ---------------------------------


@app.route("/chatbot")
def chatbot_page():
    # ✅ FIX: Use 'user_id' from session.
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    # ✅ FIX: Query chat history by the user's ID.
    history = Chat.query.filter_by(user_id=user.id).all()

    # Pass the actual user object to the template
    return render_template(
        "chatbot.html", user=user, history=history
    )


@app.route("/get_response", methods=["POST"])
def get_response():
    # ✅ FIX: Check for 'user_id'
    if "user_id" not in session:
        return jsonify({"reply": "⚠️ Please login first."})

    data = request.get_json()
    user_message = data.get("message")

    # ... (language logic) ...

    # Generate bot reply
    response = chat.send_message(user_message)
    bot_reply = response.text

    # ✅ FIX: Create chat entries using the correct user_id. This solves your storage error.
    user_chat_entry = Chat(
        user_id=session["user_id"], sender="user", message=user_message
    )
    bot_chat_entry = Chat(user_id=session["user_id"], sender="bot", message=bot_reply)

    db.session.add(user_chat_entry)
    db.session.add(bot_chat_entry)
    db.session.commit()

    return jsonify({"reply": bot_reply})


@app.route("/clear_chat")
def clear_chat():
    if "user_id" in session:
        # This logic was already correct, just confirming it.
        Chat.query.filter_by(user_id=session["user_id"]).delete()
        db.session.commit()
        flash("Chat is cleared. Scroll down to start fresh.", "success")
    return redirect(url_for("chatbot_page"))


################################----------------------------------------------------


##########################################################################################
@app.route("/tutor")
def tutor():
    return render_template("tutor.html")


##########################################################################################
# @app.route("/about")
# def about():
#     user = User.query.filter_by(username=session["username"]).first()
#     return render_template("about.html", username=user.username)


# ##########################################################################################
# @app.route("/contact")
# def contact():
#     user = User.query.filter_by(username=session["username"]).first()
#     return render_template("contact.html", username=user.username)

@app.route("/about")
def about():
    user = None
    # Check if a user is logged in by looking for user_id in the session
    if "user_id" in session:
        user = User.query.get(session["user_id"])
    # Pass the user object (which could be None) to the template
    return render_template("about.html", user=user)

# -----------------

@app.route("/contact")
def contact():
    user = None
    # Check if a user is logged in by looking for user_id in the session
    if "user_id" in session:
        user = User.query.get(session["user_id"])
    # Pass the user object (which could be None) to the template
    return render_template("contact.html", user=user)
##########################################################################################
if __name__ == "__main__":

    app.run(debug=True)
