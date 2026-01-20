import os
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot"]

def cleanup_inactive_users():
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    
    # Find users who haven't been active
    inactive_users = db["user_metadata"].find({
        "last_upload": {"$lt": thirty_days_ago}
    })

    for user in inactive_users:
        uid = user["user_id"]
        # 1. Clear their MongoDB docs
        db["raw_docs"].delete_many({"user_id": uid})
        db["queries"].delete_many({"user_id": uid})
        
        # 2. Remove their metadata
        db["user_metadata"].delete_one({"user_id": uid})
        print(f"Cleaned up data for inactive user: {uid}")

if __name__ == "__main__":
    cleanup_inactive_users()
