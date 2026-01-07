from flask import Flask, request, jsonify
from rag_engine import generate_answer
from pymongo import MongoClient
from flask_cors import CORS
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from ingest import ingest_document

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

@app.route("/")
def home():
    return "Backend running"

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json or {}
    q = data.get("question", "").strip()
    mode = data.get("mode", "Detailed")
    memory = data.get("memory", [])

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
        result = generate_answer(q, mode, memory)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    record = {
        "question": q,
        "mode": mode,
        "text": result.get("text", ""),
        "confidence": result.get("confidence", ""),
        "coverage": result.get("coverage", 0),
        "created_at": datetime.utcnow()
    }

    try:
        queries.insert_one(record)
    except Exception as e:
        print("Mongo insert failed:", e)

    return jsonify({
        "text": result.get("text", ""),
        "confidence": result.get("confidence", ""),
        "coverage": result.get("coverage", 0)
    })


@app.route("/api/history", methods=["GET"])
def history():
    try:
        data = list(queries.find({}, {"_id": 0}).sort("created_at", -1).limit(50))
        return jsonify(data)
    except Exception as e:
        return jsonify([])
    
@app.route("/api/history/<question>", methods=["GET"])
def get_history_item(question):
    item = queries.find_one({"question": {"$regex": f"^{question}$", "$options": "i"}}, {"_id": 0})
    if not item:
        return jsonify({"error": "Not found"}), 404
    return jsonify(item)

@app.route("/api/upload", methods=["POST"])
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
        print("Upload/Ingest error:", repr(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)














