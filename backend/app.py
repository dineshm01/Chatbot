from flask import Flask, request, jsonify
from rag_engine import generate_answer
from db import queries
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

client = MongoClient(os.getenv("MONGO_URI"))
db = client["rag_chatbot"]
chat_collection = db["chat_history"]


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

    # 👋 Greeting shortcut
    greetings = {"hi", "hello", "hey", "hai", "hii"}
    if q.lower() in greetings:
        return jsonify({
            "text": "Hi! 👋 How can I help you?",
            "confidence": "Greeting",
            "coverage": 0
        })

    result = generate_answer(q, mode, memory)


    record = {
        "question": q,
        "mode": mode,
        "text": result["text"],
        "confidence": result["confidence"],
        "coverage": result["coverage"],
        "created_at": datetime.utcnow()
    }

    queries.insert_one(record)

    return jsonify({
        "text": result["text"],
        "confidence": result["confidence"],
        "coverage": result["coverage"]
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
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ingest_document(filepath)

    return jsonify({"message": "File uploaded and ingested successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

