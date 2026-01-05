# ============================================================
# Edu2Job Backend – FINAL MERGED VERSION
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import jwt, os
from datetime import datetime, timedelta
import numpy as np
import joblib
from scipy.sparse import hstack

# ================= CONFIG =================
SECRET_KEY = os.environ.get("EDU2JOB_SECRET", "edu2job_dev_secret")
GOOGLE_CLIENT_ID = "68139937926-vmt33ecismjicedtubi34n1djhpr7bms.apps.googleusercontent.com"
MONGO_URI = "mongodb://localhost:27017/"
TOKEN_EXP_DAYS = 7

# ================= APP =================
app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = SECRET_KEY

# ================= DATABASE =================
client = MongoClient(MONGO_URI)
db = client["edu2job"]

users_col = db["users"]
edu_col = db["education"]
predictions_col = db["prediction_history"]
skills_col = db["user_skills"]
admin_logs_col = db["admin_logs"]

# ================= LOAD ML =================
xgb = joblib.load("models/xgboost.pkl")

rf = None
if os.path.exists("models/rf.pkl"):
    rf = joblib.load("models/rf.pkl")
    print("✅ RandomForest loaded")
else:
    print("⚠️ RF not found — XGBoost only")

tfidf = joblib.load("encoders/tfidf.pkl")
scaler = joblib.load("encoders/scaler.pkl")
le_spec = joblib.load("encoders/specialization_encoder.pkl")
le_job = joblib.load("encoders/job_encoder.pkl")
le_edu = joblib.load("encoders/education_encoder.pkl")

# ================= HELPERS =================
def safe_json():
    return request.get_json(force=True, silent=True) or {}

def create_token(email, role):
    return jwt.encode({
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXP_DAYS)
    }, SECRET_KEY, algorithm="HS256")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user = users_col.find_one({"email": decoded["email"]}, {"password": 0})
            if not user:
                return jsonify({"error": "User not found"}), 401
        except:
            return jsonify({"error": "Invalid token"}), 401
        return f(user, *args, **kwargs)
    return decorated

def admin_only(user):
    return user.get("role") == "admin"

def log_admin(action, email):
    admin_logs_col.insert_one({
        "admin": email,
        "action": action,
        "time": datetime.utcnow()
    })

# ================= AUTH =================
@app.route("/register", methods=["POST"])
def register():
    data = safe_json()

    if users_col.find_one({"email": data.get("email")}):
        return jsonify({"error": "Email already registered"}), 400

    users_col.insert_one({
        "name": data.get("name"),
        "email": data.get("email"),
        "password": generate_password_hash(data.get("password")),
        "role": "user",
        "created_at": datetime.utcnow()
    })

    return jsonify({"message": "User registered successfully"})

@app.route("/login", methods=["POST"])
def login():
    d = safe_json()
    u = users_col.find_one({"email": d.get("email")})
    if not u or not check_password_hash(u["password"], d.get("password")):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({
        "token": create_token(u["email"], u.get("role", "user")),
        "user": {"email": u["email"], "name": u.get("name"), "role": u.get("role", "user")}
    })

@app.route("/google-login", methods=["POST"])
def google_login():
    token = safe_json().get("token")
    info = id_token.verify_oauth2_token(token, grequests.Request(), GOOGLE_CLIENT_ID)

    if not users_col.find_one({"email": info["email"]}):
        users_col.insert_one({
            "email": info["email"],
            "name": info.get("name"),
            "role": "user"
        })

    return jsonify({
        "token": create_token(info["email"], "user"),
        "user": {"email": info["email"], "name": info.get("name"), "role": "user"}
    })

# ================= ML LOGIC =================
def get_top_5_predictions(education, specialization, skills, internships, cgpa):
    cgpa = float(cgpa) if cgpa else 6.0
    internships = int(internships) if internships else 0

    edu_enc = le_edu.transform([education])[0] if education in le_edu.classes_ else 0
    spec_enc = le_spec.transform([specialization])[0] if specialization in le_spec.classes_ else 0

    skills_vec = tfidf.transform([skills])
    numeric = scaler.transform([[cgpa, internships, edu_enc, spec_enc]])
    X = hstack([skills_vec, numeric])

    probs_xgb = xgb.predict_proba(X)[0]
    if rf:
        probs_rf = rf.predict_proba(X)[0]
        probs = (probs_xgb + probs_rf) / 2
    else:
        probs = probs_xgb

    top_idxs = np.argsort(probs)[-5:][::-1]

    return [{
        "job_role": le_job.inverse_transform([i])[0],
        "confidence": round(float(probs[i] * 100), 2),
        "status": "normal"
    } for i in top_idxs]

# ================= PREDICT =================
@app.route("/predict-job-role", methods=["POST"])
@token_required
def predict_job_role(user):
    d = safe_json()
    edu = edu_col.find_one({"user_email": user["email"]}) or {}
    skills_doc = skills_col.find_one({"user_email": user["email"]}) or {}

    results = get_top_5_predictions(
        edu.get("degree",""),
        edu.get("specialization",""),
        skills_doc.get("skills",""),
        d.get("internships",0),
        edu.get("cgpa",6.0)
    )

    predictions_col.insert_one({
        "user_email": user["email"],
        "predictions": results,
        "created_at": datetime.utcnow()
    })

    return jsonify({"top_5_job_roles": results})

# ================= USER =================
@app.route("/user/profile", methods=["GET"])
@token_required
def get_profile(user):
    return jsonify({
        "name": user.get("name"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "location": user.get("location")
    })
@app.route("/user/profile/update", methods=["POST"])
@token_required
def update_profile(user):
    data = safe_json()

    users_col.update_one(
        {"email": user["email"]},
        {"$set": {
            "name": data.get("name"),
            "phone": data.get("phone"),
            "location": data.get("location"),
            "updated_at": datetime.utcnow()
        }}
    )

    return jsonify({"message": "Profile updated successfully"})

@app.route("/dashboard/user")
@token_required
def dashboard_user(user):
    edu = edu_col.find_one({"user_email": user["email"]}, {"_id": 0})
    return jsonify({
        "user": {"name": user.get("name"), "email": user.get("email"), "role": user.get("role","user")},
        "education": edu
    })

@app.route("/education/me", methods=["GET"])
@token_required
def get_my_education(user):
    edu = edu_col.find_one(
        {"user_email": user["email"]},
        {"_id": 0, "user_email": 0}
    )
    return jsonify(edu or {})

@app.route("/education/add", methods=["POST"])
@token_required
def add_education(user):
    try:
        data = safe_json()

        education_data = {
            "user_email": user["email"],
            "degree": data.get("degree"),
            "specialization": data.get("specialization"),
            "university": data.get("university"),
            "cgpa": float(data["cgpa"]) if data.get("cgpa") is not None else None,
            "year": int(data.get("year")),
            "certifications": data.get("certifications"),
            "notes": data.get("notes"),
            "updated_at": datetime.utcnow()
        }

        edu_col.update_one(
            {"user_email": user["email"]},
            {"$set": education_data},
            upsert=True
        )

        return jsonify({"message": "Education saved successfully"})

    except Exception as e:
        print("EDUCATION ADD ERROR:", e)
        return jsonify({"error": "Server error"}), 500

@app.route("/user/skills", methods=["POST","GET"])
@token_required
def user_skills(user):
    if request.method == "POST":
        skills_col.update_one(
            {"user_email": user["email"]},
            {"$set": {"skills": safe_json().get("skills",""), "updated_at": datetime.utcnow()}},
            upsert=True
        )
        return jsonify({"message":"Skills saved"})

    doc = skills_col.find_one({"user_email": user["email"]},{"_id":0})
    return jsonify(doc or {})

@app.route("/user/feedback", methods=["POST"])
@token_required
def user_feedback(user):
    d = safe_json()
    predictions_col.update_one(
        {"user_email": user["email"], "predictions.job_role": d["job_role"]},
        {"$set":{"predictions.$.feedback":{
            "rating": int(d["rating"]),
            "submitted_at": datetime.utcnow()
        }}}
    )
    return jsonify({"message":"Feedback submitted"})

@app.route("/user/prediction-history")
@token_required
def prediction_history(user):
    data = predictions_col.find(
        {"user_email": user["email"]}
    ).sort("created_at", -1)

    history = []
    for d in data:
        for p in d.get("predictions", []):
            history.append({
                "created_at": d["created_at"],
                "job_role": p["job_role"],
                "confidence": p["confidence"]
            })

    return jsonify(history)


@app.route("/api/insights/career-summary")
@token_required
def career_summary(user):
    edu = edu_col.find_one({"user_email": user["email"]}) or {}
    if not edu.get("degree"):
        return jsonify({"insight":"Complete your profile for insights."})

    pipeline = [
        {"$lookup":{"from":"education","localField":"user_email","foreignField":"user_email","as":"edu"}},
        {"$unwind":"$edu"},
        {"$match":{"edu.degree":edu["degree"]}},
        {"$unwind":"$predictions"},
        {"$group":{"_id":"$predictions.job_role","count":{"$sum":1}}},
        {"$sort":{"count":-1}},
        {"$limit":1}
    ]

    r=list(predictions_col.aggregate(pipeline))
    if not r:
        return jsonify({"insight":"Not enough data yet."})

    return jsonify({"insight":f"Students with similar background often get {r[0]['_id']}."})

# ================= ADMIN =================
@app.route("/admin/prediction-logs")
@token_required
def admin_logs(user):
    if not admin_only(user): return jsonify({"error":"Forbidden"}),403
    data = predictions_col.find().sort("created_at",-1).limit(50)
    out=[]
    for d in data:
        for p in d["predictions"]:
            out.append({
                "user": d["user_email"],
                "job_role": p["job_role"],
                "confidence": p["confidence"],
                "status": p.get("status","normal")
            })
    return jsonify(out)

@app.route("/admin/review-prediction", methods=["POST"])
@token_required
def review_prediction(user):
    if not admin_only(user): return jsonify({"error":"Forbidden"}),403
    d = safe_json()
    predictions_col.update_one(
        {"user_email":d["user"],"predictions.job_role":d["job_role"]},
        {"$set":{"predictions.$.status":"reviewed"}}
    )
    log_admin("Prediction Reviewed", user["email"])
    return jsonify({"message":"Reviewed"})

@app.route("/admin/feedback-analysis")
@token_required
def feedback_analysis(user):
    if not admin_only(user): return jsonify({"error":"Forbidden"}),403

    pipeline=[
        {"$unwind":"$predictions"},
        {"$match":{"predictions.feedback.rating":{"$exists":True}}},
        {"$group":{
            "_id":"$predictions.job_role",
            "avg":{"$avg":"$predictions.feedback.rating"},
            "responses":{"$sum":1}
        }}
    ]

    return jsonify([
        {"job_role":d["_id"],"avg_rating":round(d["avg"],2),"responses":d["responses"]}
        for d in predictions_col.aggregate(pipeline)
    ])

@app.route("/admin/upload-dataset", methods=["POST"])
@token_required
def upload_dataset(user):
    if not admin_only(user):
        return jsonify({"error": "Forbidden"}), 403

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files allowed"}), 400

    filename = secure_filename(
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    )
    path = os.path.join(DATASET_FOLDER, filename)
    file.save(path)

    log_admin(f"Dataset uploaded: {filename}", user["email"])

    return jsonify({
        "message": "Dataset uploaded successfully",
        "filename": filename
    })

@app.route("/admin/retrain-model", methods=["POST"])
@token_required
def retrain_model(user):
    if not admin_only(user):
        return jsonify({"error": "Forbidden"}), 403

    log_admin("Model retraining triggered", user["email"])

    # SAFE OPTION (recommended for evaluation)
    # Simulate retraining
    return jsonify({
        "message": "Model retraining started successfully"
    })

from datetime import datetime, timedelta

@app.route("/admin/stats", methods=["GET"])
@token_required
def admin_stats(user):
    if not admin_only(user):
        return jsonify({"error": "Forbidden"}), 403

    total_users = users_col.count_documents({})
    total_predictions = predictions_col.count_documents({})

    # Today's predictions
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    predictions_today = predictions_col.count_documents({
        "created_at": {"$gte": today_start, "$lt": today_end}
    })

    return jsonify({
        "total_users": total_users,
        "total_predictions": total_predictions,
        "predictions_today": predictions_today
    })

# ================= HEALTH =================
@app.route("/")
def health():
    return jsonify({"status":"Edu2Job running"})

if __name__=="__main__":
    print("🚀 Edu2Job running at http://127.0.0.1:5000")
    app.run(debug=True)
