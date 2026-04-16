#!/usr/bin/env bash
# Build the FastAPI Docker image and push it to Amazon ECR.
#
# Usage:
#   chmod +x aws/ecr_push_api.sh
#   ./aws/ecr_push_api.sh <account-id> <region>
#
# Example:
#   ./aws/ecr_push_api.sh 123456789012 us-east-1

set -euo pipefail

ACCOUNT_ID="${1:?Usage: $0 <account-id> <region>}"
REGION="${2:?}"
REPO_NAME="vsp-api"
TAG="latest"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

echo "==> Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Creating ECR repository (if it doesn't exist)..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" \
    --region "${REGION}" > /dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" \
       --region "${REGION}"

echo "==> Building FastAPI Docker image..."
docker build -f docker/Dockerfile.api -t "${REPO_NAME}:${TAG}" .

echo "==> Tagging image..."
docker tag "${REPO_NAME}:${TAG}" "${IMAGE_URI}"

echo "==> Pushing to ECR: ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Done! API image pushed to:"
echo "  ${IMAGE_URI}"
echo ""
echo "Next: run ./aws/deploy_api.sh ${ACCOUNT_ID} ${REGION} <your-key-pair>"