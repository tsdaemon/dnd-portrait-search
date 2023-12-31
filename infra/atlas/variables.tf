variable "mongodb_atlas_org_id" {
  type        = string
  description = "MongoDB Atlas Organization ID"
}

variable "mongodb_atlas_root_username" {
  type        = string
  description = "MongoDB Atlas user root name"
  sensitive   = true
}

variable "mongodb_atlas_root_password" {
  type        = string
  description = "MongoDB Atlas user root password"
  sensitive   = true
}

variable "mongodb_atlas_ip_access_cidr_block_azure" {
  type        = string
  description = "Allowed IP addresses for Azure environment"
}

variable "mongodb_atlas_ip_access_cidr_block_local" {
  type        = string
  description = "Allowed IP addresses for local environment"
}
