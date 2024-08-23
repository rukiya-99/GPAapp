from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional


app = FastAPI()

#load the train model
model = joblib.load('student_gpa_model.pkl')

#define the data structure for the input
class StudentData (BaseModel):
    Gender: str
    Age: int
    StudyHoursPerWeek: int
    AttendanceRate: float
    Major: str
    PartTimeJob: str
    ExtraCurricularActivities:str

@app.post("/predict/")
def predict (data:StudentData):

    #convert input data to dataframe
    input_data = pd.DataFrame([data.dict()])

    #make prediction using the model
    try:
        prediction = model.predict(input_data)
        return{"GPA":prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
if __name__ == "Main":
        import uvicorn
        uvicorn.run(app, host = "0.0.0.0",port=8000)
