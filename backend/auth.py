import jwt
import datetime
import os

SECRET = os.getenv("JWT_SECRET", "supersecretkey")

def create_token(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def verify_token(token):
    return jwt.decode(token, SECRET, algorithms=["HS256"], options={"require": ["exp"]})
