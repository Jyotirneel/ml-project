from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from src.inference import Predictor
from src.breed_info import BREED_INFO

app = FastAPI(title="Cattle Breed Recognition API")

predictor = Predictor(model_path="models/cattle_breed_model.pt", breed_info=BREED_INFO)

# Security constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.get("/")
def root():
    return {"message": "Cattle Breed Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Security: Validate file extension
    filename = file.filename or ""
    extension = filename.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File extension not allowed. Allowed: {ALLOWED_EXTENSIONS}")

    image_bytes = await file.read()

    # Security: Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    results = predictor.predict(image_bytes)
    return JSONResponse(content=results)

@app.get("/breeds")
def breeds():
    return BREED_INFO

