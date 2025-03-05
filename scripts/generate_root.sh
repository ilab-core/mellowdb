#!/bin/bash

# Create the creds directory
mkdir -p creds

# Generate Root CA
echo "Enter the Common Name (CN) for the Root CA:"
read -p "CN (e.g., RootCA): " CN

echo "Enter the number of days the Root CA certificate should be valid:"
read -p "Days (e.g., 3650): " DAYS

openssl genrsa -out creds/ca.key 2048
openssl req -x509 -new -nodes -key creds/ca.key -sha256 -days "$DAYS" -out creds/ca.crt -subj "/CN=${CN}"

echo "Root CA generated as creds/ca.crt and creds/ca.key, valid for $DAYS days."
