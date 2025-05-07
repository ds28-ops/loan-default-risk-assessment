resource "openstack_compute_instance_v2" "nodes" {
  count             = 3
  name              = "node${count.index + 1}-cloud-project43"
  image_id          = var.image_id
  flavor_id         = var.flavor_id
  key_pair          = var.key_pair
  availability_zone = var.availability_zone

  security_groups = var.security_groups

  network {
    name = var.private_network
  }

  network {
    name = var.public_network
  }

  metadata = {
    environment = "project43"
  }
}
