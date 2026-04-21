import io
import json
import logging
import os
import flask
import torch
from PIL import Image

logger = logging.getLogger(__name__)
app = flask.Flask(__name__)

# Load model once at startup
model = None

def get_model():
    global model
    if model is None:
        import sys
        sys.path.insert(0, "/opt/ml/code")
        from inference.inference import model_fn
        model = model_fn("/opt/ml/model")
    return model

@app.route("/ping", methods=["GET"])
def ping():
    health = get_model() is not None
    status = 200 if health else 503
    return flask.Response(
        response=json.dumps({"status": "healthy" if health else "unhealthy"}),
        status=status,
        mimetype="application/json"
    )

@app.route("/invocations", methods=["POST"])
def invocations():
    from inference.inference import input_fn, predict_fn, output_fn
    content_type = flask.request.content_type or "application/octet-stream"
    data = input_fn(flask.request.data, content_type)
    prediction = predict_fn(data, get_model())
    result, mimetype = output_fn(prediction, "application/json")
    return flask.Response(response=result, status=200, mimetype=mimetype)