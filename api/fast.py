from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"Yohoo":"Finally last recap !"}

@app.get("/predict")
def predict(
    pickup_datetime, 
    lon1, 
    lat1, 
    lon2, 
    lat2, 
    passenger_count=1):
    
    X_pred = pd.DataFrame(dict(
        key=["key"],
        pickup_datetime=[pickup_datetime],
        pickup_longitude=[float(lon1)],
        pickup_latitude=[float(lat1)],
        dropoff_longitude=[float(lon2)],
        dropoff_latitude=[float(lat2)],
        passenger_count=[int(passenger_count)] 
    ))
    
    pipeline = joblib.load('model.joblib')
    
    y_pred = float(pipeline.predict(X_pred)[0])
    
    return {"the prediction" : y_pred}