from fastapi import FastAPI 
from pydantic import BaseModel
import pandas as pd 
import joblib 
import os

app =FastAPI()

__file__=""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_file = os.path.join(BASE_DIR, "models", "heart_pipline.pkl")
heart_pipeline=joblib.load(model_file)
model_file = os.path.join(BASE_DIR, "models", "diabetes_pipline.pkl")
diabetes_pipeline=joblib.load(model_file)
model_file = os.path.join(BASE_DIR, "models", "cancer_pipline.pkl")
cancer_pipeline=joblib.load(model_file)

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalac: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
class DiabetesDiseaseInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
class BreastCancerDiseaseInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst:float
    texture_worst:float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst:float
    points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float
@app.get("/")
def home():
    return "Disease Prediction API is running"
@app.post("/predict/heart")
def predict_heart(data: HeartDiseaseInput):
    input_df=pd.DataFrame([data.model_dump()])
    prediction=heart_pipeline.predict(input_df)[0]
    return {"prediction":int(prediction),
            "daignosis":"Heart Disease Detected" if prediction ==1 else "No Heart Disease"}
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesDiseaseInput):
    input_df=pd.DataFrame([data.model_dump()])
    prediction=diabetes_pipeline.predict(input_df)[0]
    return {"prediction":int(prediction),
            "daignosis":"Diabertes Disease Detected" if prediction ==1 else "No Diabetes Disease"}
@app.post("/predict/cancer")
def predict_breast_cancer(data: BreastCancerDiseaseInput):
    input_df=pd.DataFrame([data.model_dump()])
    prediction=cancer_pipeline.predict(input_df)[0]
    return {"prediction":int(prediction),
            "daignosis":"Breast Cancer Disease Detected" if prediction ==1 else "No Breast Cancer Disease"}

    


