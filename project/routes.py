# /project/routes.py

from flask import (
    Blueprint,
    request,
    redirect,
    render_template,
    url_for,
    jsonify,
    flash,
    session,
    g,
    current_app,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from sqlalchemy import desc
from functools import wraps
from datetime import datetime
import io
import re
import json
import concurrent.futures

from flask_mail import Message 
from .extensions import mail   

# Import from our own project files
from .models import db, User, Chat, MedicalReport, BlogPost
from .services import (
    get_predicted_value,
    get_disease_details,
    symptoms_dict,
    ai_prediction_model,
    chat_session,
)

# Create a Blueprint
main_bp = Blueprint("main", __name__)

# --- DECORATORS ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            # flash("⚠️ Please login to access this page.", "warning")
            return redirect(url_for("main.login"))
        
        user = User.query.get(session["user_id"])
        
        if user is None:
            session.clear()
            # flash("Your session is invalid. Please log in again.", "danger")
            return redirect(url_for("main.login"))
        
        g.user = user
        return f(*args, **kwargs)
    return decorated_function

# --- ERROR HANDLERS ---
@main_bp.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("File too large! Maximum allowed size is 2 MB.", "danger")
    return redirect(url_for("main.dashboard"))

# --- HELPER FUNCTIONS ---
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]

# --- AUTHENTICATION & CORE ROUTES ---
@main_bp.route("/")
def home1():
    return render_template("index.html")

@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            flash(f"✅ Welcome, {username}! Login successful!", "success")
            return redirect(url_for("main.home"))
        else:
            # flash("Invalid username or password.", "danger")
            return redirect(url_for("main.login"))
    return render_template("index.html", error="Invalid username or password.")

@main_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if User.query.filter_by(username=username).first():
            # flash("Username already exists!", "danger")
            return render_template("register.html", error="User already exists!")

        if password != confirm_password:
            # flash("Passwords do not match!", "danger")
            return render_template("register.html", error="Passwords do not match!")

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session["user_id"] = new_user.id
        flash(f"✅ Welcome, {new_user.username}! Your account has been created.", "success")
        return redirect(url_for("main.home"))
    return render_template("register.html")

@main_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    # flash("You have been logged out.", "success")
    return redirect(url_for("main.login"))

# --- USER-FACING PAGES ---
@main_bp.route("/home")
@login_required
def home():
    return render_template("home.html", user=g.user)

@main_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        # Handle profile updates
        new_username = request.form.get("username")
        if new_username and new_username != g.user.username:
            if User.query.filter_by(username=new_username).first():
                flash("❌ Username already taken!", "danger")
                return redirect(url_for("main.profile"))
            g.user.username = new_username
        
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")
        if old_password and new_password:
            if not g.user.check_password(old_password):
                flash("❌ Old password is incorrect!", "danger")
                return redirect(url_for("main.profile"))
            g.user.set_password(new_password)

        db.session.commit()
        flash("✅ Profile updated successfully!", "success")
        return redirect(url_for("main.profile"))
    return render_template("profile.html", user=g.user)

@main_bp.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    db.session.delete(g.user)
    db.session.commit()
    session.clear()
    # flash("Your account has been deleted.", "success")
    return redirect(url_for("main.register"))

# --- STORAGE ROUTES ---
@main_bp.route("/dashboard")
@login_required
def dashboard():
    reports = MedicalReport.query.filter_by(user_id=g.user.id).order_by(desc(MedicalReport.uploaded_at)).all()
    return render_template("dashboard.html", user=g.user, reports=reports)

@main_bp.route("/upload", methods=["POST"])
@login_required
def upload():
    if "report" not in request.files:
        flash("No file part in the form.", "warning")
        return redirect(url_for("main.dashboard"))
    file = request.files["report"]
    if file.filename == "":
        flash("No file was selected.", "warning")
        return redirect(url_for("main.dashboard"))
    if not allowed_file(file.filename):
        flash("File type not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    safe_name = secure_filename(file.filename)
    data = file.read()
    report = MedicalReport(
        original_name=safe_name,
        mimetype=file.mimetype,
        size=len(data),
        user_id=g.user.id,
        data=data,
    )
    db.session.add(report)
    db.session.commit()
    flash("✅ Report uploaded successfully!", "success")
    return redirect(url_for("main.dashboard"))

@main_bp.route("/files/<int:report_id>")
@login_required
def download(report_id):
    report = MedicalReport.query.get_or_404(report_id)
    if report.user_id != g.user.id:
        abort(403)
    return send_file(
        io.BytesIO(report.data),
        as_attachment=True,
        download_name=report.original_name,
        mimetype=report.mimetype,
    )

@main_bp.route("/files/<int:report_id>/delete", methods=["POST"])
@login_required
def delete_file(report_id):
    report = MedicalReport.query.get_or_404(report_id)
    if report.user_id != g.user.id:
        abort(403)
    db.session.delete(report)
    db.session.commit()
    flash("🗑️ Report deleted successfully!", "success")
    return redirect(url_for("main.dashboard"))

# --- PREDICTION ROUTES ---
@main_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        symptoms = request.form.getlist("symptoms")
        if not symptoms:
            flash("Please select at least one symptom.", "warning")
            return redirect(url_for('main.predict'))
        
        predicted_disease = get_predicted_value(symptoms)
        details = get_disease_details(predicted_disease)
        
        return render_template(
            "predict.html",
            user=g.user,
            symptoms_list=list(symptoms_dict.keys()),
            prediction_result=True,
            name=request.form.get("patient_name"),
            age=request.form.get("patient_age"),
            gender=request.form.get("patient_gender"),
            symptoms=symptoms,
            predicted_disease=predicted_disease,
            report_time=datetime.now().strftime("%d-%m-%Y %H:%M"),
            dis_desc=details['description'],
            dis_prec=details['precautions'],
            dis_medi=details['medications'],
            dis_diet=details['diets'],
            dis_wrk=details['workouts'],
        )
    return render_template("predict.html", user=g.user, symptoms_list=list(symptoms_dict.keys()))

# --- AI & CHATBOT ROUTES ---
@main_bp.route("/ai_predict", methods=["GET", "POST"])
@login_required
def ai_predict():
    if request.method == "POST":
        user_prompt = f"Name: {request.form.get('name')}, Age: {request.form.get('age')}, Gender: {request.form.get('gender')}, Symptoms: {request.form.get('symptoms')}"
        try:
            response = ai_prediction_model.generate_content(user_prompt)
            cleaned_text = re.sub(r"^```json|```$", "", response.text.strip()).strip()
            prediction_data = json.loads(cleaned_text)
            return render_template("ai_predict.html", user=g.user, result=prediction_data, report_time=datetime.now().strftime("%d-%m-%Y %H:%M"))
        except Exception as e:
            current_app.logger.error(f"AI Prediction error: {e}")
            flash("⚠️ AI service is unavailable. Please try again later.", "danger")
            return redirect(url_for("main.ai_predict"))
    return render_template("ai_predict.html", user=g.user)

@main_bp.route("/chatbot")
@login_required
def chatbot_page():
    history = Chat.query.filter_by(user_id=g.user.id).all()
    return render_template("chatbot.html", user=g.user, history=history)

@main_bp.route("/get_response", methods=["POST"])
@login_required
def get_response():
    user_message = request.get_json().get("message")
    response = chat_session.send_message(user_message)
    bot_reply = response.text
    
    db.session.add(Chat(user_id=g.user.id, sender="user", message=user_message))
    db.session.add(Chat(user_id=g.user.id, sender="bot", message=bot_reply))
    db.session.commit()
    return jsonify({"reply": bot_reply})

@main_bp.route("/clear_chat")
@login_required
def clear_chat():
    Chat.query.filter_by(user_id=g.user.id).delete()
    db.session.commit()
    flash("Chat is cleared. Scroll down to start fresh.", "success")
    return redirect(url_for("main.chatbot_page"))

# --- BLOG & STATIC PAGES ---
@main_bp.route('/blog')
def blog_index():
    user = User.query.get(session["user_id"]) if "user_id" in session else None
    post_id = request.args.get('post_id', type=int)
    if post_id:
        single_post = BlogPost.query.get_or_404(post_id)
        return render_template('blog_index.html', post=single_post, user=user)
    else:
        all_posts = BlogPost.query.order_by(desc(BlogPost.created_at)).all()
        return render_template('blog_index.html', posts=all_posts, user=user)

@main_bp.route("/about")
def about():
    user = User.query.get(session["user_id"]) if "user_id" in session else None
    return render_template("about.html", user=user)

# @main_bp.route("/contact")
# def contact():
#     user = User.query.get(session["user_id"]) if "user_id" in session else None
#     return render_template("contact.html", user=user)

@main_bp.route("/contact", methods=["GET", "POST"]) # <-- Add methods
def contact():
    if request.method == "POST":
        # Get form data
        name = request.form.get("name")
        email = request.form.get("email")
        message_body = request.form.get("message")

        # Basic validation
        if not name or not email or not message_body:
            flash("All fields are required.", "danger")
            return redirect(url_for("main.contact"))

        # Create the email message
        msg = Message(
            subject=f"New Contact Form Submission from {name}",
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[current_app.config['MAIL_USERNAME']]  # Send to yourself
        )
        msg.body = f"From: {name} <{email}>\n Message : {message_body}"

        try:
            # Send the email
            mail.send(msg)
            flash("Thank you for your message! It has been sent.", "success")
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")

        return redirect(url_for("main.contact"))

    # GET request just displays the page
    user = User.query.get(session["user_id"]) if "user_id" in session else None
    return render_template("contact.html", user=user)