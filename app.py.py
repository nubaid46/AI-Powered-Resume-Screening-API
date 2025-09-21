# app.py

import os
import re
import json
import datetime
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)

# NLP & parsing libs
import pdfplumber
import docx

import pytesseract
from PIL import Image

# ---- ADD THIS LINE: path to your tesseract executable ----
# Make sure this exact path matches where you installed Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESSERACT_AVAILABLE = True

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional: sentence-transformers (semantic similarity)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDER = None
    EMBEDDINGS_AVAILABLE = False

# -------- config & setup --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["JWT_SECRET_KEY"] = "replace-this-with-a-secure-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "resumes.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
jwt = JWTManager(app)

# load spaCy model (make sure you've run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

ALLOWED_EXT = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}

# ---------- DB models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    raw_text = db.Column(db.Text)
    parsed_json = db.Column(db.Text)  # store parsed fields as JSON
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# create DB tables if not exist
with app.app_context():
    db.create_all()

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            page_text = p.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("Error extracting from image:", e)
        return ""

def extract_text(path, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    if ext == "docx":
        return extract_text_from_docx(path)
    if ext in {"png", "jpg", "jpeg"}:
        return extract_text_from_image(path)
    if ext == "txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""

SKILLS_DB = {"python","flask","django","sql","postgresql","mysql","docker","aws",
             "nlp","machine learning","pandas","numpy","scikit-learn","tensorflow",
             "pytorch","git","rest","api","jwt","celery"}

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

def parse_basic_fields(text):
    data = {"emails": [], "phones": [], "name": None, "skills": [], "total_experience_years": None}
    data["emails"] = list(set(EMAIL_RE.findall(text)))
    data["phones"] = list(set(PHONE_RE.findall(text)))
    m = re.search(r'(\d+)\s*\+?\s*(?:years|yrs)', text.lower())
    if m:
        try:
            data["total_experience_years"] = int(m.group(1))
        except:
            data["total_experience_years"] = None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            data["name"] = ent.text
            break
    found = set()
    text_low = text.lower()
    for s in SKILLS_DB:
        if re.search(r'\b' + re.escape(s) + r'\b', text_low):
            found.add(s)
    data["skills"] = sorted(list(found))
    return data

def compute_scores(resume_text, jd_text, resume_parsed=None, jd_required_skills=None, jd_min_experience=None):
    jd_text = jd_text or ""
    resume_text = resume_text or ""
    if jd_required_skills is None:
        jd_required_skills = []
        jd_low = jd_text.lower()
        for s in SKILLS_DB:
            if re.search(r'\b' + re.escape(s) + r'\b', jd_low):
                jd_required_skills.append(s)
    resume_skills = set(resume_parsed.get("skills", [])) if resume_parsed else set()
    required_set = set(jd_required_skills)
    matched = resume_skills & required_set
    skill_score = (len(matched) / len(required_set)) if required_set else 1.0

    resume_exp = (resume_parsed.get("total_experience_years") if resume_parsed else None)
    exp_score = 1.0
    if jd_min_experience:
        if resume_exp is None:
            exp_score = 0.0
        else:
            exp_score = min(1.0, resume_exp / jd_min_experience)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        v = vectorizer.fit_transform([resume_text, jd_text])
        tfidf_sim = float(cosine_similarity(v[0], v[1])[0][0])
    except Exception:
        tfidf_sim = 0.0

    embed_sim = None
    if EMBEDDINGS_AVAILABLE:
        try:
            emb = EMBEDDER.encode([resume_text, jd_text])
            import numpy as np
            a, b = emb[0], emb[1]
            cos = float(np.dot(a, b) / ( (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10 ))
            embed_sim = cos
        except Exception:
            embed_sim = None

    w_skill = 0.5
    w_exp = 0.2
    w_tfidf = 0.2
    w_embed = 0.1

    score = w_skill * skill_score + w_exp * exp_score + w_tfidf * tfidf_sim
    total_w = w_skill + w_exp + w_tfidf + (w_embed if embed_sim is not None else 0)
    if embed_sim is not None:
        score = score + w_embed * embed_sim

    final_score = score / total_w if total_w > 0 else 0.0

    return {
        "final_score": round(float(final_score), 4),
        "breakdown": {
            "skill_score": round(float(skill_score), 4),
            "experience_score": round(float(exp_score), 4),
            "tfidf_similarity": round(float(tfidf_sim), 4),
            "embedding_similarity": (round(float(embed_sim), 4) if embed_sim is not None else None)
        },
        "matched_skills": sorted(list(matched)),
        "required_skills": sorted(list(required_set))
    }

# ---------- Auth endpoints ----------
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"msg":"username & password required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"msg":"user exists"}), 400
    u = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(u)
    db.session.commit()
    return jsonify({"msg":"registered"}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    u = User.query.filter_by(username=username).first()
    if not u or not check_password_hash(u.password_hash, password):
        return jsonify({"msg":"bad username/password"}), 401
    access_token = create_access_token(identity=u.id)
    return jsonify({"access_token": access_token})

# ---------- Upload resume ----------
@app.route("/resume/upload", methods=["POST"])
@jwt_required()
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"msg":"file part required"}), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify({"msg":"no selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"msg":"file type not allowed"}), 400
    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    text = extract_text(save_path, filename) or ""
    parsed = parse_basic_fields(text)
    user_id = get_jwt_identity()
    r = Resume(user_id=user_id, filename=filename, raw_text=text, parsed_json=json.dumps(parsed))
    db.session.add(r)
    db.session.commit()
    return jsonify({"msg":"uploaded","resume_id": r.id, "parsed": parsed})

# ---------- Match endpoint (resume id or upload) ----------
@app.route("/match", methods=["POST"])
@jwt_required()
def match_resume():
    data = request.form or {}
    resume_id = data.get("resume_id")
    jd_text = data.get("job_description") or (request.json and request.json.get("job_description"))
    jd_min_experience = data.get("min_experience")
    try:
        jd_min_experience = int(jd_min_experience) if jd_min_experience is not None else None
    except:
        jd_min_experience = None

    resume_text = ""
    parsed = {}
    if resume_id:
        r = Resume.query.filter_by(id=int(resume_id)).first()
        if not r:
            return jsonify({"msg":"resume not found"}), 404
        resume_text = r.raw_text or ""
        parsed = json.loads(r.parsed_json or "{}")
    elif 'file' in request.files:
        f = request.files['file']
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            resume_text = extract_text(path, filename) or ""
            parsed = parse_basic_fields(resume_text)
        else:
            return jsonify({"msg":"invalid file"}), 400
    else:
        return jsonify({"msg":"provide resume_id or upload file"}), 400

    scores = compute_scores(resume_text, jd_text or "", resume_parsed=parsed, jd_min_experience=jd_min_experience)
    return jsonify({"scores": scores, "parsed": parsed})

# ---------- serve uploaded files (for debugging) ----------
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
