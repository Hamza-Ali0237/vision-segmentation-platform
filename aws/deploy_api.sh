#!/usr/bin/env bash
# Deploy the FastAPI container to a t3.micro EC2 instance.
#
# This script:
#   1. Creates a Security Group allowing HTTP (8000) and SSH (22)
#   2. Launches a t3.micro instance with the API Docker image
#   3. Prints the public IP and the API URL
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker image pushed to ECR (run ecr_push_api.sh first)
#   - A key pair created in EC2 (for SSH access)
#
# Usage:
#   chmod +x aws/deploy_api.sh
#   ./aws/deploy_api.sh <account-id> <region> <key-pair-name>
#
# Example:
#   ./aws/deploy_api.sh 123456789012 us-east-1 my-keypair

set -euo pipefail

ACCOUNT_ID="${1:?Usage: $0 <account-id> <region> <key-pair-name>}"
REGION="${2:?}"
KEY_PAIR="${3:?}"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/vsp-api:latest"
SG_NAME="vsp-api-sg"
INSTANCE_TYPE="t3.micro"   # free tier eligible

echo "==> Creating Security Group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "VSP API security group" \
    --region "${REGION}" \
    --query 'GroupId' --output text 2>/dev/null \
  || aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --region "${REGION}" \
    --query 'SecurityGroups[0].GroupId' --output text)

echo "Security Group: ${SG_ID}"

# Allow SSH and API port
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" --region "${REGION}" \
    --protocol tcp --port 22   --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" --region "${REGION}" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 2>/dev/null || true

echo "==> Getting latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text --region "${REGION}")

echo "AMI: ${AMI_ID}"

# User data script — installs Docker, logs into ECR, runs the container
USER_DATA=$(cat <<EOF
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Log into ECR
aws ecr get-login-password --region ${REGION} \
  | docker login --username AWS --password-stdin \
    ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Pull and run the API container
docker pull ${IMAGE_URI}
docker run -d \
  --restart always \
  -p 8000:8000 \
  -e AWS_REGION=${REGION} \
  --name vsp-api \
  ${IMAGE_URI}
EOF
)

echo "==> Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_PAIR}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile Name=EC2SageMakerRole \
    --user-data "${USER_DATA}" \
    --region "${REGION}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vsp-api}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: ${INSTANCE_ID}"
echo "==> Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${REGION}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "========================================="
echo "  API deployed!"
echo "  Instance ID : ${INSTANCE_ID}"
echo "  Public IP   : ${PUBLIC_IP}"
echo ""
echo "  Wait ~2 minutes for Docker to start, then:"
echo "  API URL  : http://${PUBLIC_IP}:8000"
echo "  Docs     : http://${PUBLIC_IP}:8000/docs"
echo "  Health   : http://${PUBLIC_IP}:8000/health"
echo ""
echo "  SSH access:"
echo "  ssh -i ~/.ssh/${KEY_PAIR}.pem ec2-user@${PUBLIC_IP}"
echo "========================================="