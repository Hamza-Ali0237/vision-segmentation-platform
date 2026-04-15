# -----------------------------------------------------------------------
# SageMaker Training Container — CXR Segmentation Platform
#
# Build:
#   docker build -f docker/Dockerfile -t cxr-segmentation:latest .
#
# Push to ECR (replace placeholders):
#   aws ecr get-login-password --region us-east-1 \
#     | docker login --username AWS \
#       --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
#   docker tag cxr-segmentation:latest \
#     ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cxr-segmentation:latest
#   docker push \
#     ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cxr-segmentation:latest
# -----------------------------------------------------------------------

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

LABEL maintainer="vision-segmentation-platform"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source
COPY . .

# SageMaker expects the training script at the path passed via
# estimator(entry_point=...).  We set PYTHONPATH so package imports work.
ENV PYTHONPATH=/opt/ml/code

# Default entrypoint — SageMaker overrides this with the entry_point arg
ENTRYPOINT ["python", "train.py"]