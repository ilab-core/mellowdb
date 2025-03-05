#!/bin/bash

# Create the creds directory
mkdir -p creds

# Generate Server Key and Certificate with SANs
echo "Enter the Common Name (CN) for the Server Certificate:"
read -p "CN (e.g., server.domain.com): " CN

echo "Enter the Subject Alternative Names (SANs) for the Server Certificate (comma-separated):"
read -p "SANs (e.g., DNS:localhost,IP:127.0.0.1): " SANs

echo "Enter the number of days the Server certificate should be valid:"
read -p "Days (e.g., 365): " DAYS

openssl genrsa -out creds/server.key 2048
openssl req -new -key creds/server.key -out creds/server.csr -subj "/CN=${CN}"

# Create SANs file
cat > creds/server.ext <<EOF
subjectAltName = ${SANs}
EOF

openssl x509 -req -in creds/server.csr -CA creds/ca.crt -CAkey creds/ca.key -CAcreateserial -out creds/server.crt -days "$DAYS" -sha256 -extfile creds/server.ext

echo "Server certificate generated as creds/server.crt and creds/server.key, valid for $DAYS days."
