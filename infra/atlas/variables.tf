variable "org_id" {
  type        = string
  description = "MongoDB Atlas Organization ID"
}

variable "root_username" {
  type      = string
  sensitive = true
}

variable "root_password" {
  type        = string
  description = "MongoDB Atlas Database root password"
  sensitive   = true
}

variable "ip_access_cidr_block_azure" {
  type        = string
  description = "Allowed IP addresses for Azure environment"
}

variable "ip_access_cidr_block_local" {
  type        = string
  description = "Allowed IP addresses for local environment"
}
