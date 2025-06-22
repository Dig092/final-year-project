from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

def get_auth_token():
    return os.getenv("AUTH_TOKEN","afcd6dd3-5657-4331-88f8-521f6569235d")

def user_auth_dependency(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials:
        raise HTTPException(status_code=403, detail="Credentials are required")
    
    token = credentials.credentials
    expected_token = get_auth_token()
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")