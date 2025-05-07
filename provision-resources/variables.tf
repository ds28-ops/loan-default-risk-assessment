variable "image_id" {
  description = "ID of the image to use"
  type        = string
  default     = "96d9c658-6540-4796-ae64-54d8ac6c45f8"
}

variable "flavor_id" {
  description = "Flavor ID"
  type        = string
  default     = "3"
}

variable "key_pair" {
  description = "Key pair to use for SSH access"
  type        = string
  default     = "id_rsa_chameleon"
}

variable "availability_zone" {
  description = "Availability zone"
  type        = string
  default     = "nova"
}

variable "private_network" {
  description = "Name of the private network"
  type        = string
  default     = "private_cloud_net_project43"
}

variable "public_network" {
  description = "Name of the public/shared network"
  type        = string
  default     = "sharednet1"
}

variable "security_groups" {
  description = "List of security groups to assign"
  type        = list(string)
  default     = ["default", "allow-ssh", "allow-http-80"]
}
