import uvicorn
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import io
from PIL import Image
import os
import shutil
from fastapi.responses import RedirectResponse
from leaf import test_leaf
from chat import get_completion_from_messages,collect_messages_text
# from chat import collect_messages_text
app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
context = []
templates = Jinja2Templates(directory="templates")
data = {}
class Message(BaseModel):
    content: str
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message:Message):
    user_message = message.content
    print(context)
    response = collect_messages_text(user_message,context)
    return {"message": response}

@app.post("/upload")
async def upload_image(request: Request,image: UploadFile = File(...)):
    if image is not None :
        # data["name"] = name 
        # Saving the uploaded image to the specified directory
        image_path = os.path.join('static', image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        d = test_leaf(image_path)
        
        data.update(d)
        context.append({"role": "system", "content": f"""Consider you are a chatbot.
        The details of the user are as follows and\
        The user provides an leaf image and the details from the leaf image are as follows:\
        {str(data)}.\
        Now act as a chatbot and answer questions asked by the user.\
        First, give the user the leaf report, then ask the user whether he/she has any questions.\
        Answer the questions wisely in short form.\
        Use the name of the user to interact.\
        If the user asks questions out of context - Simply warn him.\
        Now act as a chatbot and answer questions asked by the user.\
        First of all, greet the user with his name and show the result given in the leaf data."""})
        
        # Return a response with the processed image (and any other data)
        
        return templates.TemplateResponse("index.html", {"request": request, "image_url":f'{image_path}',"path": '/static/output_image.jpg',"name":d["Name of leaf"]})
    return templates.TemplateResponse("index.html", {"request": request, "error": "No image selected."})



if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)


   #apt-get install -y libgl1-mesa-dev
   #sudo apt-get install libgl1-mesa-glx
   #pip install python-multipart
   #python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl