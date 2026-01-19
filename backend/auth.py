import jwt
import datetime
import os

# This line looks for JWT_SECRET in Railway variables. 
# If not found, it uses the fallback "supersecretkey" (not recommended for production).
SECRET = os.getenv("JWT_SECRET", "supersecretkey")

def create_token(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    # Uses the SECRET variable to sign the token
    return jwt.encode(payload, SECRET, algorithm="HS256")

def verify_token(token):
    # Uses the SAME SECRET to decode and verify the token
    return jwt.decode(
        token,
        SECRET,
        algorithms=["HS256"],
        options={"require": ["exp"]}
    )
