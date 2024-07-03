from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
model = pickle.load(open("RandomForest.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("Home.html", {"request": request})

@app.get("/crop", response_class=HTMLResponse)
async def home(request: Request):
    query_params = request.query_params
    # Access specific query parameters by key
    crop_type = query_params.get("type")
    print(crop_type)
    return templates.TemplateResponse(f"/crops/{crop_type}.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("Login.html", {"request": request})


@app.post("/predict")
async def predict_crop(request: Request, N: float = Form(...), P: float = Form(...), K: float = Form(...),
                       temperature: float = Form(...), humidity: float = Form(...), ph: float = Form(...),
                       rainfall: float = Form(...)):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    print("Input Features:", input_features)
    result = model.predict(input_features)
    predicted_crop = result[0]
    return {"predicted_crop": predicted_crop}