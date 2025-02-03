import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

modelo = joblib.load('./Model/random_forest_model.pkl')
scaler = joblib.load('./Model/scaler.pkl')
LabelEncoder = joblib.load('./Model/label_encoder.pkl')

@app.get('/predecir/')
async def predecir(min_salary: int, max_salary: int, has_company_logo: bool, has_questions: bool, employment_type: int, required_experience: int, required_education: int):
    nuevo = pd.DataFrame({'min_salary': [min_salary], 
                          'max_salary': [max_salary], 
                          'has_company_logo': [has_company_logo], 
                          'has_questions': [has_questions], 
                          'employment_type': [employment_type], 
                          'required_experience': [required_experience], 
                          'required_education': [required_education]})
    nuevo_scaled = scaler.transform(nuevo)
    prediccion = modelo.predict(nuevo_scaled)
    prediccion = int(prediccion[0])
    return {'prediction': prediccion}