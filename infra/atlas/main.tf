#
# Example Terraform Config to create a
# MongoDB Atlas Shared Tier Project, Cluster,
# Database User and Project IP Whitelist Entry
#
# First step is to create a MongoDB Atlas account
# https://docs.atlas.mongodb.com/tutorial/create-atlas-account/
#
# Then create an organization and programmatic API key
# https://docs.atlas.mongodb.com/tutorial/manage-organizations
# https://docs.atlas.mongodb.com/tutorial/manage-programmatic-access
#
# Terraform MongoDB Atlas Provider Documentation
# https://www.terraform.io/docs/providers/mongodbatlas/index.html
# Terraform 0.14+, MongoDB Atlas Provider 0.9.1+

#
# Configure the MongoDB Atlas Provider
#
terraform {
  required_providers {
    mongodbatlas = {
      source = "mongodb/mongodbatlas"
    }
  }
}

provider "mongodbatlas" {
}

#
# Create a Project
#
resource "mongodbatlas_project" "project" {
  name   = "portrait-search"
  org_id = var.mongodb_atlas_org_id
}
#
# Create a Shared Tier Cluster
#
resource "mongodbatlas_cluster" "cluster" {
  project_id                   = mongodbatlas_project.project.id
  name                         = "portrait-search-cluster"
  provider_name                = "TENANT"
  backing_provider_name        = "AZURE"
  provider_region_name         = "EUROPE_WEST"
  provider_instance_size_name  = "M0"
  mongo_db_major_version       = "6.0"
  auto_scaling_disk_gb_enabled = "false"
}

#
# Create an Atlas Admin Database User
#
resource "mongodbatlas_database_user" "root_user" {
  username           = var.mongodb_atlas_root_username
  password           = var.mongodb_atlas_root_password
  project_id         = mongodbatlas_project.project.id
  auth_database_name = "admin"

  roles {
    role_name     = "readWrite"
    database_name = var.mongodb_atlas_database_name
  }

  roles {
    role_name     = "readAnyDatabase"
    database_name = "admin"
  }

  scopes {
    name = mongodbatlas_cluster.cluster.name
    type = "CLUSTER"
  }
}

#
# Create an IP Accesslist
#
resource "mongodbatlas_project_ip_access_list" "local" {
  project_id = mongodbatlas_project.project.id
  cidr_block = var.mongodb_atlas_ip_access_cidr_block_local
  comment    = "Local IP Addresses"
}

resource "mongodbatlas_project_ip_access_list" "azure" {
  project_id = mongodbatlas_project.project.id
  cidr_block = var.mongodb_atlas_ip_access_cidr_block_azure
  comment    = "Azure IP Addresses"
}
