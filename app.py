from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]  

app = FastAPI(title='Heart Disease Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],  
   allow_headers=["*"]  
)

model = load(pathlib.Path('model/heart-disease-v1.joblib'))

class InputData(BaseModel):
    age:int=64
    sex:int=1
    cp:int=3
    trestbps:int=120
    chol:int=267
    fbs:int=0
    restecg:int=0
    thalach:int=99
    exang:int=1
    oldpeak:float=1.8
    slope:int=1
    ca:int=2
    thal:int=2
    
class OutputData(BaseModel):
    score:float=0.80318881046519

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = pd.DataFrame([data.dict()])
    result = model.predict_proba(model_input)[:,-1].item()

    return {'score':result}
