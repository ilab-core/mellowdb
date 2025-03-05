#!/bin/bash

# Create the creds directory
mkdir -p creds

# Generate Client Key and Certificate
echo "Enter the Common Name (CN) for the Client Certificate:"
read -p "CN (e.g., client): " CN

echo "Enter the number of days the Client certificate should be valid:"
read -p "Days (e.g., 365): " DAYS

openssl genrsa -out creds/client.key 2048
openssl req -new -key creds/client.key -out creds/client.csr -subj "/CN=${CN}"
openssl x509 -req -in creds/client.csr -CA creds/ca.crt -CAkey creds/ca.key -CAcreateserial -out creds/client.crt -days "$DAYS" -sha256

echo "Client certificate generated as creds/client.crt and creds/client.key, valid for $DAYS days."
