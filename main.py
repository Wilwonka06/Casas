from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib 

model = joblib.load('modelo_precio_casas.pkl')
columnas = joblib.load('columnas_modelo_precio_casas.pkl')

app = FastAPI(title="Prediccion precio de Ventas")

# Modelo de entrada
class InputData(BaseModel):
    superficie: float
    habitaciones: int 
    antiguedad: int
    ubicacion: int
    
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de precios de casas"}

@app.post("/predict/")
def predict(data: InputData):
    x_new = pd.DataFrame([[data.superficie, data.habitaciones, data.antiguedad, data.ubicacion]], columns=columnas)
    prediccion = model.predict(x_new)
    return {"prediccion": float(prediccion[0])}