#!/usr/bin/env bash
# One-time GCP setup for the Text-to-Alpha model service (Cloud Run, us-central1).
# Run step by step; every command creates or changes a resource. Free tier only:
#   Cloud Run request-based billing (180k vCPU-s, 360k GiB-s, 2M requests / month free),
#   Artifact Registry 0.5 GB free (cleanup policy keeps the 3 newest images).
# No JSON service-account keys: GitHub Actions authenticates via Workload Identity Federation.
set -euo pipefail

PROJECT_ID=text-to-alpha
PROJECT_NUMBER=348925486034
REGION=us-central1
AR_REPO=t2a
SERVICE=t2a-service
GITHUB_REPO=WenxueWu0216/Text-to-Alpha
GITHUB_REPO_ID=1445869322          # immutable; a renamed/recreated repo can't inherit access
POOL=github
PROVIDER=text-to-alpha
DEPLOY_SA=t2a-deployer@${PROJECT_ID}.iam.gserviceaccount.com
RUNTIME_SA=t2a-runtime@${PROJECT_ID}.iam.gserviceaccount.com

# 1. APIs. iam + sts are needed for service accounts and the WIF token exchange.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com \
  iam.googleapis.com sts.googleapis.com --project "$PROJECT_ID"

# 2. Artifact Registry repo + cleanup policy (stay under the 0.5 GB free tier).
gcloud artifacts repositories create "$AR_REPO" --project "$PROJECT_ID" --location "$REGION" \
  --repository-format docker --description "Text-to-Alpha model service images"
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --project "$PROJECT_ID" --location "$REGION" \
  --policy infra/ar-cleanup-policy.json --no-dry-run

# 3. Service accounts. The runtime SA has NO roles (the service calls no Google APIs);
#    it replaces the default compute SA, which has project Editor.
gcloud iam service-accounts create t2a-runtime --project "$PROJECT_ID" --display-name "t2a Cloud Run runtime (no roles)"
gcloud iam service-accounts create t2a-deployer --project "$PROJECT_ID" --display-name "t2a GitHub Actions deployer"

# 4. Deployer permissions, as narrow as Cloud Run allows:
#    - push images to this one repo only
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --project "$PROJECT_ID" --location "$REGION" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/artifactregistry.writer
#    - deploy revisions as the runtime SA, and only that SA
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --project "$PROJECT_ID" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/iam.serviceAccountUser
#    - create/update Cloud Run services, split traffic. run.developer cannot change IAM, so CI
#      can never make a private service public (or the reverse). Scoped to the service in step 7.

# 5. Workload Identity Federation: GitHub OIDC tokens from pushes to main of this repo only.
gcloud iam workload-identity-pools create "$POOL" --project "$PROJECT_ID" --location global \
  --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --project "$PROJECT_ID" \
  --location global --workload-identity-pool "$POOL" --display-name "Text-to-Alpha main" \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref" \
  --attribute-condition "assertion.repository_id == '${GITHUB_REPO_ID}' && assertion.ref == 'refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --project "$PROJECT_ID" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository_id/${GITHUB_REPO_ID}"

# 6. First deploy, by a human: creates the service and makes it public (allUsers invoker).
#    Later deploys come from CI and never touch IAM.
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE}:bootstrap"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet          # local docker config only
docker build --platform linux/amd64 -f service/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
gcloud run deploy "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --service-account "$RUNTIME_SA" --allow-unauthenticated \
  --cpu 1 --memory 1Gi --concurrency 8 --timeout 30 --min-instances 0 --max-instances 2 \
  --port 8080 --set-env-vars T2A_NUM_THREADS=1 --cpu-throttling --no-cpu-boost

# 7. Now that the service exists, grant the deployer run.developer on THIS service only.
gcloud run services add-iam-policy-binding "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/run.developer

# 8. GitHub repo variables (none are secrets: WIF needs no key, and the provider name and SA
#    email are useless without a token minted for this repo's main branch).
gh variable set GCP_PROJECT_ID --body "$PROJECT_ID"
gh variable set GCP_REGION --body "$REGION"
gh variable set GAR_REPO --body "$AR_REPO"
gh variable set CLOUD_RUN_SERVICE --body "$SERVICE"
gh variable set GCP_DEPLOY_SA --body "$DEPLOY_SA"
gh variable set GCP_RUNTIME_SA --body "$RUNTIME_SA"
gh variable set GCP_WIF_PROVIDER --body "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
