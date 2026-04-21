#!/bin/bash
if [ "$1" = "train" ]; then
    python train.py
elif [ "$1" = "serve" ]; then
    gunicorn \
        --timeout 60 \
        --workers 1 \
        --worker-class sync \
        --bind 0.0.0.0:8080 \
        "inference.serving:app"
else
    exec "$@"
fi