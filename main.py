from fastapi import (
    FastAPI,
    Request,
    Form,
    Depends,
    HTTPException,
    status,
    Cookie,
)
from fastapi import FastAPI, HTTPException, Depends, Cookie
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse,HTMLResponse
from pydantic import BaseModel
from pymongo import MongoClient
import bcrypt

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["fastapi_users"]
users_collection = db["users"]

# OAuth2 Password Flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


#  Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

# User Model
class User(BaseModel):
    username: str
    password: str

# Register New User
@app.post("/register")
async def register(user: User):
    # Check if user already exists
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    # Hash the password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    # Store user details in MongoDB
    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}

# Login
@app.post("/login")
async def login(response: RedirectResponse, form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"username": form_data.username})
    if user and bcrypt.checkpw(form_data.password.encode('utf-8'), user['password']):
        # If login successful, set a cookie or session token to manage authentication
        response.set_cookie(key="user", value=user['username'])
        return response
    raise HTTPException(status_code=401, detail="Invalid username or password")

# Main Page
@app.get("/main")
async def main_page(user: str = Cookie(None), token: str = Depends(oauth2_scheme)):
    if user:
        return {"message": f"Welcome, {user}!"}
    raise HTTPException(status_code=401, detail="User not authenticated")

# Serve HTML Files
@app.get("/", response_class=HTMLResponse)
def home(request: Request, error: str = None):
   return templates.TemplateResponse("register.html",{"request": request, "error": error} )

@app.get("/login")
async def login(request: Request, error: str = None):
    return templates.TemplateResponse("login.html",{"request": request, "error": error} )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
