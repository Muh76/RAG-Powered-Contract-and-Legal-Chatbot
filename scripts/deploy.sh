#!/usr/bin/env bash
# Deploy Legal Chatbot backend to Google Cloud Run.
# Prereqs: gcloud auth login, gcloud config set project PROJECT_ID
# Secrets (DATABASE_URL, JWT_SECRET_KEY, OPENAI_API_KEY, etc.) must be set in Cloud Run
#   after deploy (Console → Edit & deploy → Variables) or via Secret Manager + --set-secrets.

set -e

# --- Config (override via env or edit) ---
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-legal-chatbot-api}"
REPO_NAME="${ARTIFACT_REGISTRY_REPO:-legal-chatbot}"
IMAGE_NAME="${IMAGE_NAME:-legal-chatbot-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: PROJECT_ID not set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=============================================="
echo "Deploying to Cloud Run"
echo "  Project:  $PROJECT_ID"
echo "  Region:   $REGION"
echo "  Service:  $SERVICE_NAME"
echo "  Image:    $FULL_IMAGE"
echo "=============================================="

# Enable APIs (idempotent)
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com run.googleapis.com --project="$PROJECT_ID"

# Create Artifact Registry repo if missing
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  echo "Creating Artifact Registry repository: $REPO_NAME"
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID"
fi

# Build and push with Cloud Build (no local Docker required)
echo "Building and pushing image..."
gcloud builds submit \
  --tag "$FULL_IMAGE" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  .

# Deploy to Cloud Run (non-secret env vars only; set secrets in Console or via --set-secrets)
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image="$FULL_IMAGE" \
  --platform=managed \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --allow-unauthenticated \
  --set-env-vars="PORT=8080,ENVIRONMENT=production,LOG_LEVEL=INFO,LOG_FORMAT=json" \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10

echo "=============================================="
echo "Deployment complete."
echo "Set required secrets in Cloud Run Console (Edit & deploy → Variables & Secrets):"
echo "  DATABASE_URL, JWT_SECRET_KEY (or JWT_SECRET), OPENAI_API_KEY, REDIS_URL (optional)"
echo "=============================================="
gcloud run services describe "$SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)'
