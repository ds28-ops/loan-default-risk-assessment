output "node_ips" {
  description = "Floating IPs of the nodes"
  value = [for instance in openstack_compute_instance_v2.nodes : instance.access_ip_v4]
}

output "node_names" {
  description = "Names of the created instances"
  value = [for instance in openstack_compute_instance_v2.nodes : instance.name]
}
