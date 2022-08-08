import json

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib

from mangum import Mangum

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        # load serialized model
        self.model_config = config
        model = joblib.load(config['classifier']['serialized_model_path'])
        # construct prediction pipeline
        featurizer = TransformerFeaturizer(
            dim=config['featurizer']['sentence_transformer_embedding_dim'],
            sentence_transformer_model=SentenceTransformer(
                f"sentence-transformers/{config['featurizer']['sentence_transformer_model']}"
            )
        )
        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
        ])
        self.classes = model.classes_

    def predict_proba(self, model_input: str) -> dict:
        prediction = self.pipeline.predict_proba([model_input])
        classes_to_probs = dict(zip(self.classes, prediction[0].tolist()))
        return classes_to_probs

    def predict_label(self, model_input: str) -> dict:
        prediction = self.pipeline.predict([model_input])
        return prediction[0]

app = FastAPI()
data = {}
global_config = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "logs.out"
    }
}

@app.on_event("startup")
def startup_event():
    # Read configs
    # create the prediction model instance
    data['model'] = NewsCategoryClassifier(global_config['model'])
    data['logger'] = open(global_config['service']["log_destination"], 'w', encoding='utf-8')
    logger.info("Setup completed")

@app.on_event("shutdown")
def shutdown_event():
    # clean up
    data['logger'].close()
    logger.info("Shutting down application")


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = data['model'].predict_proba(request.description)
    to_log = {
        'timestamp': '...',
        'request': request.dict(),
        'prediction': prediction,
        'latency': '...'
    }
    logger.info(to_log)
    data['logger'].write(json.dumps(to_log) + "\n")
    data['logger'].flush()
    return {"prediction": prediction}

@app.get("/")
def read_root():
    return {"Hello": "World"}


def handler(event, context):
    asgi_handler = Mangum(app)
    # Call the instance with the event arguments
    response = asgi_handler(event, context)
    return response
