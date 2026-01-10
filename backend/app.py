from flask import Flask, request, jsonify
from rag_engine import generate_answer
from pymongo import MongoClient
from flask_cors import CORS
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from ingest import ingest_document
from bson import ObjectId
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import bcrypt
from auth import create_token, verify_token



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chatbot"]
queries = db["queries"]
users = db["users"]

def require_auth(fn):
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = auth.split(" ")[1]
        try:
            payload = verify_token(token)
            request.user_id = payload["user_id"]
        except:
            return jsonify({"error": "Invalid token"}), 401

        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


@app.route("/")
def home():
    return "Backend running"

@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if users.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user = users.insert_one({"email": email, "password": hashed})

    token = create_token(user.inserted_id)
    return jsonify({"token": token})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode(), user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user["_id"])
    return jsonify({"token": token})


@app.route("/api/ask", methods=["POST"])
@require_auth
def ask():
    data = request.json or {}
    q = data.get("question", "").strip()
    mode = data.get("mode", "Detailed")
    memory = data.get("memory", [])
    strict = data.get("strict", False)


    if not q:
        return jsonify({"error": "Question is required"}), 400

    greetings = {"hi", "hello", "hey", "hai", "hii"}
    if q.lower() in greetings:
        return jsonify({
            "text": "Hi! 👋 How can I help you?",
            "confidence": "Greeting",
            "coverage": 0
        })

    try:
        result = generate_answer(q, mode, memory, strict)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    display_text = result.get("display_text", result["text"])

    record = {
        "user_id": request.user_id,
        "question": q,
        "mode": mode,
        "text": display_text,
        "confidence": result.get("confidence", ""),
        "coverage": result.get("coverage", 0),
        "sources": result.get("sources", []),
        "chunks": result.get("chunks", []),
        "feedback": None,
        "bookmarked": False,
        "created_at": datetime.utcnow()
    }

    try:
        res = queries.insert_one(record)
        record_id = str(res.inserted_id)
        
    except Exception as e:
        print("Mongo insert failed:", e)

    return jsonify({
        "id": record_id,
        "text": display_text,
        "confidence": result["confidence"],
        "coverage": result["coverage"],
        "chunks": result.get("chunks", []),
        "sources": result.get("sources", []),
        "debug": result.get("debug", {})
    })

@app.route("/api/history", methods=["GET"])
@require_auth
def get_history():
    data = list(
        queries.find({"user_id": request.user_id})
        .sort("created_at", -1)
        .limit(50)
    )

    # convert ObjectId to string for JSON
    for item in data:
        item["_id"] = str(item["_id"])
        item["feedback"] = item.get("feedback")
        item["bookmarked"] = item.get("bookmarked", False)



        # normalize coverage format
        if isinstance(item.get("coverage"), int):
            item["coverage"] = {
                "grounded": item["coverage"],
                "general": 100 - item["coverage"]
            }

    return jsonify(data)

@app.route("/api/history/id/<id>", methods=["GET"])
@require_auth
def get_history_by_id(id):
    try:
        item = queries.find_one({"_id": ObjectId(id), "user_id": request.user_id}, {"_id": 0})
        if not item:
            return jsonify({"error": "Not found"}), 404
        return jsonify(item)
    except:
        return jsonify({"error": "Invalid id"}), 400
    
@app.route("/api/history/<question>", methods=["GET"])
@require_auth
def get_history_item(question):
    item = queries.find_one({
        "question": {"$regex": f"^{question}$", "$options": "i"},
        "user_id": request.user_id
    }, {"_id": 0})
    
    if not item:
        return jsonify({"error": "Not found"}), 404

    if isinstance(item.get("coverage"), int):
        item["coverage"] = {"grounded": item["coverage"], "general": 100 - item["coverage"]}

    return jsonify(item)

@app.route("/api/history/search", methods=["GET"])
@require_auth
def search_history():
    q = request.args.get("q", "").strip()

    if not q:
        return jsonify([])

    results = list(queries.find({
        "user_id": request.user_id,
        "$or": [
            {"question": {"$regex": q, "$options": "i"}},
            {"text": {"$regex": q, "$options": "i"}},
            {"sources.source": {"$regex": q, "$options": "i"}}
        ]
    }).sort("created_at", -1).limit(50))

    for item in results:
        item["_id"] = str(item["_id"])

    return jsonify(results)

@app.route("/api/history/<question>", methods=["DELETE"])
@require_auth
def delete_history_item(question):
    result = queries.delete_one({
        "question": {"$regex": f"^{question}$", "$options": "i"},
        "user_id": request.user_id
    })
    if result.deleted_count == 0:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"message": "Deleted"}), 200

@app.route("/api/history", methods=["DELETE"])
@require_auth
def delete_all_history():
    queries.delete_many({})
    return jsonify({"message": "All history deleted"}), 200
    
@app.route("/api/upload", methods=["POST"])
@require_auth
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        ingest_document(filepath)

        return jsonify({"message": "Uploaded"}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": repr(e)}), 500

@app.route("/api/feedback", methods=["POST"])
@require_auth
def save_feedback():
    data = request.json or {}
    msg_id = data.get("id")
    feedback = data.get("feedback")

    if not msg_id or feedback not in ["up", "down"]:
        return jsonify({"error": "Invalid input"}), 400

    queries.update_one(
        {"_id": ObjectId(msg_id)},
        {"$set": {"feedback": feedback}}
    )

    return jsonify({"message": "Feedback saved", "feedback": feedback}), 200

@app.route("/api/bookmark", methods=["POST"])
@require_auth
def bookmark():
    data = request.json or {}
    msg_id = data.get("id")
    value = data.get("value")

    if not msg_id or value not in [True, False]:
        return jsonify({"error": "Invalid input"}), 400

    queries.update_one(
        {"_id": ObjectId(msg_id)},
        {"$set": {"bookmarked": value}}
    )

    return jsonify({"message": "Bookmark updated", "value": value})


@app.route("/api/analytics", methods=["GET"])
@require_auth
def analytics():
    uid = request.user_id
    
    total = queries.count_documents({"user_id": uid})
    helpful = queries.count_documents({"user_id": uid, "feedback": "up"})
    wrong = queries.count_documents({"user_id": uid, "feedback": "down"})
    bookmarked = queries.count_documents({"user_id": uid, "bookmarked": True})

    most_questions = list(
        queries.aggregate([
            {"$match": {"user_id": uid}},
            {"$group": {"_id": "$question", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ])
    )

    most_sources = list(
        queries.aggregate([
            {"$match": {"user_id": uid}},
            {"$unwind": "$sources"},
            {"$group": {"_id": "$sources.source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ])
    )

    hallucination_rate = round((wrong / total) * 100, 2) if total else 0

    return jsonify({
        "total": total,
        "helpful": helpful,
        "wrong": wrong,
        "bookmarked": bookmarked,
        "hallucination_rate": hallucination_rate,
        "top_questions": most_questions,
        "top_sources": most_sources
    })

@app.route("/api/export", methods=["GET"])
@require_auth
def export_history_pdf():
    items = list(queries.find({"user_id": request.user_id}).sort("created_at", 1))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    for item in items:
        elements.append(Paragraph(f"<b>Q:</b> {item['question']}", styles["Normal"]))
        elements.append(Paragraph(f"<b>A:</b> {item['text']}", styles["Normal"]))

        if item.get("sources"):
            srcs = ", ".join(s["source"] for s in item["sources"])
            elements.append(Paragraph(f"<b>Sources:</b> {srcs}", styles["Normal"]))

        if item.get("feedback"):
            elements.append(Paragraph(f"<b>Feedback:</b> {item['feedback']}", styles["Normal"]))

        if item.get("bookmarked"):
            elements.append(Paragraph("<b>⭐ Bookmarked</b>", styles["Normal"]))

        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)

    return buffer.getvalue(), 200, {
        "Content-Type": "application/pdf",
        "Content-Disposition": "attachment; filename=chat_history.pdf"
    }
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


