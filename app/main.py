from fastapi import FastAPI
from app.schemas import PatientAnalysis
import pandas as pd
from fastapi.encoders import jsonable_encoder
from app.pipelines import regr_classifier
from sklearn.exceptions import NotFittedError

app = FastAPI()
clf = regr_classifier.LRClassifier()


@app.post("/predict/")
async def make_prediction(analysis: PatientAnalysis):
    json_obj = jsonable_encoder(analysis)
    data = pd.DataFrame(json_obj, index=[0])

    try:
        res = clf.predict(data).tolist()
        return {"predicted_class": res[0]}
    except NotFittedError:
        return {"message": "Model not fitted. You need first train model"}


@app.get("/")
async def root():
    return {"message": "Hello. Please follow to /docs link"}
