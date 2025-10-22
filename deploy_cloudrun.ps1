<#
deploy_cloudrun.ps1

Usage:
  1. Save this file in your project root.
  2. Open PowerShell (do not activate your Python venv for these gcloud steps).
  3. cd to project directory and run: .\deploy_cloudrun.ps1
#>

param(
  [string]$ProjectID = "cobalt-column-472507-k4",
  [string]$Region = "asia-south1",
  [string]$Bucket = "cobalt-column-472507-k4-models",
  [string]$Service = "yolo-api",
  [string]$Image = "",
  [string]$SA_Name = "yolo-reader",
  [int]$MemoryGB = 2,
  [int]$Cpu = 1
)

function ExitOnError($msg) {
  Write-Host "ERROR: $msg" -ForegroundColor Red
  exit 1
}

# Set default image if not provided
if ([string]::IsNullOrEmpty($Image)) {
  $Image = "gcr.io/$ProjectID/$Service:latest"
}

Write-Host "Using Project: $ProjectID, Region: $Region"
gcloud config set project $ProjectID
if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to set gcloud project to $ProjectID." }

# Ask whether to generate API key
$choice = Read-Host "Do you want the script to generate a random API key for you? (y/N)"
if ($choice -match '^(y|Y)') {
  $API_KEY = -join ((33..126) | Get-Random -Count 40 | ForEach-Object {[char]$_})
  Write-Host "Generated API_KEY: $API_KEY"
} else {
  $API_KEY = Read-Host -Prompt "Enter a strong API key (keep this secret)"
  if ([string]::IsNullOrEmpty($API_KEY)) { ExitOnError "No API key provided." }
}

# Find local model (default location)
$localModelCandidate = Join-Path -Path (Get-Location) -ChildPath "runs\vehicle_damage_yolov8m\weights\best.pt"
if (Test-Path $localModelCandidate) {
  $localModel = $localModelCandidate
} else {
  Write-Host "Default model not found at: $localModelCandidate" -ForegroundColor Yellow
  $alt = Read-Host "Enter path to your best.pt (or press Enter to abort)"
  if ([string]::IsNullOrEmpty($alt)) { ExitOnError "Model file not found. Place best.pt at runs/... or provide correct path." }
  if (-not (Test-Path $alt)) { ExitOnError "Provided path does not exist: $alt" }
  $localModel = $alt
}

Write-Host "Model path to upload: $localModel"

# 1) Create bucket if not exists
$gsPath = "gs://$Bucket"
Write-Host "Checking bucket $gsPath ..."
& gsutil ls -b $gsPath > $null 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "Creating bucket: $gsPath"
  & gsutil mb -l $Region $gsPath
  if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to create bucket $gsPath. Try a different (globally unique) name." }
} else {
  Write-Host "Bucket exists: $gsPath"
}

# 2) Upload model
Write-Host "Uploading model to $gsPath/best.pt ..."
& gsutil cp $localModel "$gsPath/best.pt"
if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to upload model to $gsPath/best.pt" }

# 3) Create service account if missing
$SA_Email = "$SA_Name@$ProjectID.iam.gserviceaccount.com"
Write-Host "Checking service account $SA_Email ..."
$existingSA = gcloud iam service-accounts list --filter="email:$SA_Email" --format="value(email)"
if ([string]::IsNullOrEmpty($existingSA)) {
  Write-Host "Creating service account: $SA_Email"
  gcloud iam service-accounts create $SA_Name --display-name "YOLO model reader"
  if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to create service account $SA_Email" }
} else {
  Write-Host "Service account already exists: $SA_Email"
}

# 4) Grant storage.objectViewer to the service account (project-level, simple)
Write-Host "Granting roles/storage.objectViewer to $SA_Email ..."
gcloud projects add-iam-policy-binding $ProjectID --member="serviceAccount:$SA_Email" --role="roles/storage.objectViewer"
if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to grant roles/storage.objectViewer to $SA_Email" }

# 5) Backup local best.pt (local backups folder)
$backupDir = Join-Path -Path (Get-Location) -ChildPath "backups"
if (-not (Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir | Out-Null }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupName = "best_before_${timestamp}.pt"
Copy-Item $localModel (Join-Path $backupDir $backupName)
if ($LASTEXITCODE -ne 0) { ExitOnError "Failed to create local backup of model." }
Write-Host "Local backup created: $backupDir\$backupName"

# 6) Build & push container with Cloud Build
Write-Host "Starting Cloud Build for image: $Image ..."
gcloud builds submit --tag $Image
if ($LASTEXITCODE -ne 0) { ExitOnError "Cloud Build failed. Inspect output and fix errors (requirements, Dockerfile)." }

# 7) Deploy to Cloud Run
Write-Host "Deploying to Cloud Run: service=$Service region=$Region ..."
gcloud run deploy $Service --image $Image --region $Region --platform managed `
  --allow-unauthenticated `
  --memory ${MemoryGB}Gi --cpu $Cpu --max-instances 1 --concurrency 1 `
  --service-account $SA_Email `
  --set-env-vars "MODEL_BUCKET=$gsPath,MODEL_FILENAME=best.pt,API_KEY=$API_KEY" --quiet
if ($LASTEXITCODE -ne 0) { ExitOnError "Cloud Run deployment failed." }

# 8) Print service URL & API key
$serviceUrl = gcloud run services describe $Service --region $Region --format "value(status.url)"
Write-Host "Deployment complete! Service URL:" -ForegroundColor Green
Write-Host $serviceUrl
Write-Host ""
Write-Host "Use header 'x-api-key: <your API_KEY>' when calling /predict. Your API_KEY (copy now):" -ForegroundColor Yellow
Write-Host $API_KEY -ForegroundColor Cyan

Write-Host "All done." -ForegroundColor Green
