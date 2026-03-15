# GroundTruth — Terraform Infrastructure-as-Code
# Deploys the complete stack on Google Cloud
# Bonus points: automated IaC deployment

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# =============================================
# Variables
# =============================================
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "google_api_key" {
  description = "Gemini API Key"
  type        = string
  sensitive   = true
}

# =============================================
# Provider
# =============================================
provider "google" {
  project = var.project_id
  region  = var.region
}

# =============================================
# Enable APIs
# =============================================
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
  ])

  project = var.project_id
  service = each.key

  disable_on_destroy = false
}

# =============================================
# Artifact Registry
# =============================================
resource "google_artifact_registry_repository" "groundtruth" {
  location      = var.region
  repository_id = "groundtruth"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# =============================================
# Cloud Run Service
# =============================================
resource "google_cloud_run_v2_service" "groundtruth" {
  name     = "groundtruth"
  location = var.region

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/groundtruth/groundtruth:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_API_KEY"
        value = var.google_api_key
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.groundtruth,
  ]
}

# =============================================
# IAM: Allow public access
# =============================================
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.groundtruth.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================
# Outputs
# =============================================
output "service_url" {
  description = "GroundTruth service URL"
  value       = google_cloud_run_v2_service.groundtruth.uri
}

output "project_id" {
  value = var.project_id
}
