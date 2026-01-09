from flask import Flask, request, jsonify
from rag_engine import generate_answer
from pymongo import MongoClient
from flask_cors import CORS
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from ingest import ingest_document
from bson import ObjectId

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

    display_text = result["text"]  # raw text only  

    record = {
        "question": q,
        "mode": mode,
        "text": display_text,
        "confidence": result.get("confidence", ""),
        "coverage": result.get("coverage", 0),
        "sources": result.get("sources", []),
        "chunks": result.get("chunks", []),
        "feedback": None,
        "created_at": datetime.utcnow()
    }

    try:
        queries.insert_one(record)
    except Exception as e:
        print("Mongo insert failed:", e)

    return jsonify({
        "text": display_text,
        "confidence": result["confidence"],
        "coverage": result["coverage"],
        "chunks": result.get("chunks", []),
        "sources": result.get("sources", []),
        "debug": result.get("debug", {})
    })

@app.route("/api/history", methods=["GET"])
def get_history():
    data = list(
        queries.find()
        .sort("created_at", -1)
        .limit(50)
    )

    # convert ObjectId to string for JSON
    for item in data:
        item["_id"] = str(item["_id"])
        item["feedback"] = item.get("feedback")


        # normalize coverage format
        if isinstance(item.get("coverage"), int):
            item["coverage"] = {
                "grounded": item["coverage"],
                "general": 100 - item["coverage"]
            }

    return jsonify(data)

@app.route("/api/history/id/<id>", methods=["GET"])
def get_history_by_id(id):
    try:
        item = queries.find_one({"_id": ObjectId(id)}, {"_id": 0})
        if not item:
            return jsonify({"error": "Not found"}), 404
        return jsonify(item)
    except:
        return jsonify({"error": "Invalid id"}), 400
    
@app.route("/api/history/<question>", methods=["GET"])
def get_history_item(question):
    item = queries.find_one({"question": {"$regex": f"^{question}$", "$options": "i"}}, {"_id": 0})
    if not item:
        return jsonify({"error": "Not found"}), 404

    if isinstance(item.get("coverage"), int):
        item["coverage"] = {"grounded": item["coverage"], "general": 100 - item["coverage"]}

    return jsonify(item)

@app.route("/api/history/<question>", methods=["DELETE"])
def delete_history_item(question):
    result = queries.delete_one({"question": {"$regex": f"^{question}$", "$options": "i"}})
    if result.deleted_count == 0:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"message": "Deleted"}), 200

@app.route("/api/history", methods=["DELETE"])
def delete_all_history():
    queries.delete_many({})
    return jsonify({"message": "All history deleted"}), 200
    
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": repr(e)}), 500

@app.route("/api/feedback", methods=["POST"])
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

        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

































