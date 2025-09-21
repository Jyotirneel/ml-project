from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inference import Predictor
from src.breed_info import BREED_INFO

app = FastAPI(title="Cattle Breed Recognition API")

predictor = Predictor(model_path="models/cattle_breed_model.pt", breed_info=BREED_INFO)

@app.get("/")
def root():
    return {"message": "Cattle Breed Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = predictor.predict(image_bytes)
    return JSONResponse(content=results)

@app.get("/breeds")
def breeds():
    return BREED_INFO

