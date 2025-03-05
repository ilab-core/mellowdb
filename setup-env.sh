#!/bin/bash

touch /mellow-db/.env

echo "GCP_SERVICE_ACCOUNT='${GCP_SERVICE_ACCOUNT}'" >> /mellow-db/.env
echo "MELLOW_HOST='${MELLOW_HOST}'" >> /mellow-db/.env
echo "MELLOW_PORT='${MELLOW_PORT}'" >> /mellow-db/.env
echo "MELLOW_DATA_DIR='${MELLOW_DATA_DIR}'" >> /mellow-db/.env
echo "MELLOW_SERVER_LOG_LEVEL='${MELLOW_SERVER_LOG_LEVEL}'" >> /mellow-db/.env
echo "PROJECT_ID='${PROJECT_ID}'" >> /mellow-db/.env
