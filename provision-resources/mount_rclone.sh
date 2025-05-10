#!/bin/bash
set -e

echo "Installing rclone..."
curl -s https://rclone.org/install.sh | sudo bash

echo "Enabling user_allow_other in fuse.conf..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

echo "Creating rclone config directory..."
mkdir -p ~/.config/rclone

echo "Writing rclone credentials..."
cat <<EOF > ~/.config/rclone/rclone.conf
[chi_uc]
type = swift
user_id = bab37d1c3536b5a9d505d39097b873711095368a708c729e8ffc595b524adc9c
application_credential_id = 0e8fc5e5c3054e71a51ee63457badeca
application_credential_secret = nbSGHM1Gjb65NfNVBauLibjSqDZ2dBY8aXTJBmAM3_nrLyQINj9GSjiDzDZV4Lmct07hJKm3yTvdTbdOp_MfIQ
auth = https://chi.uc.chameleoncloud.org:5000/v3
region = CHI@UC

[chi_tacc]
type = swift
user_id = 4b39694b789aa923464ead3d22e2ec88df69021623ced5e7b3e7269cf8fb4e75
application_credential_id = 5c335b442b424ad6a90dfd929b5c38e8
application_credential_secret = Ei-4YyxD0-I6HSn66yo5Z99qBwLwfKOfRKPdraHqjiMJK4amXAMaM2_ZQPMjQ7vTKnLH6wdWsOAAkKP6njoa6w
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

echo "Creating /mnt/object and setting permissions..."
sudo rm -rf /mnt/object
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

echo "Mounting rclone (read-only)..."
rclone mount chi_tacc:object-persist-project43 /mnt/object --read-only --allow-other --daemon

echo "Mount completed at /mnt/object"
