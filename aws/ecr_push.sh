#!/usr/bin/env bash
# Build the training Docker image and push it to Amazon ECR.
#
# Usage:
#   chmod +x aws/ecr_push.sh
#   ./aws/ecr_push.sh <account-id> <region> <repo-name>
#
# Example:
#   ./aws/ecr_push.sh 123456789012 us-east-1 cxr-segmentation

set -euo pipefail

ACCOUNT_ID="${1:?Usage: $0 <account-id> <region> <repo-name>}"
REGION="${2:?}"
REPO_NAME="${3:-cxr-segmentation}"
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

echo "==> Building Docker image..."
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "${REPO_NAME}:${TAG}" --load .

echo "==> Tagging image..."
docker tag "${REPO_NAME}:${TAG}" "${IMAGE_URI}"

echo "==> Pushing to ECR: ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Done! Copy this URI into training/configs/base.yaml -> aws.ecr_image_uri:"
echo "  ${IMAGE_URI}"