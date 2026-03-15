#!/bin/bash
# GroundTruth — Google Cloud Run Deployment Script
# This script automates the full deployment pipeline.
# Bonus: Infrastructure-as-Code for hackathon extra points.

set -euo pipefail

# =============================================
# Configuration
# =============================================
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="groundtruth"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:?Set GOOGLE_API_KEY}"

echo "=========================================="
echo " GroundTruth Deployment"
echo " Project: ${PROJECT_ID}"
echo " Region:  ${REGION}"
echo "=========================================="

# =============================================
# Step 1: Enable required GCP APIs
# =============================================
echo "[1/5] Enabling Google Cloud APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

# =============================================
# Step 2: Build container image
# =============================================
echo "[2/5] Building container image..."
cd "$(dirname "$0")/.."

gcloud builds submit \
    --tag "${IMAGE_NAME}:latest" \
    --project="${PROJECT_ID}" \
    --quiet

# =============================================
# Step 3: Deploy to Cloud Run
# =============================================
echo "[3/5] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_NAME}:latest" \
    --platform=managed \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --allow-unauthenticated \
    --port=8080 \
    --memory=1Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=5 \
    --timeout=300 \
    --set-env-vars="GOOGLE_API_KEY=${GOOGLE_API_KEY},GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --quiet

# =============================================
# Step 4: Get service URL
# =============================================
echo "[4/5] Fetching service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format='value(status.url)')

echo ""
echo "=========================================="
echo " Deployment Complete!"
echo " URL: ${SERVICE_URL}"
echo "=========================================="

# =============================================
# Step 5: Health check
# =============================================
echo "[5/5] Running health check..."
HEALTH_RESPONSE=$(curl -s "${SERVICE_URL}/api/health")
echo "Health: ${HEALTH_RESPONSE}"
echo ""
echo "GroundTruth is live and ready."
