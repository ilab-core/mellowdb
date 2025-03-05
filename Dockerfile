FROM python:3.11-bookworm

WORKDIR /mellow-db
COPY . /mellow-db/

ARG GCP_SERVICE_ACCOUNT
ENV GCP_SERVICE_ACCOUNT ${GCP_SERVICE_ACCOUNT}
ARG MELLOW_HOST
ENV MELLOW_HOST ${MELLOW_HOST}
ARG MELLOW_PORT
ENV MELLOW_PORT ${MELLOW_PORT}
ARG MELLOW_DATA_DIR
ENV MELLOW_DATA_DIR ${MELLOW_DATA_DIR}
ARG PROJECT_ID
ENV PROJECT_ID ${PROJECT_ID}
ARG TRIGGER_NAME
ENV TRIGGER_NAME ${TRIGGER_NAME}
ARG MELLOW_SERVER_LOG_LEVEL
ENV MELLOW_SERVER_LOG_LEVEL ${MELLOW_SERVER_LOG_LEVEL}

ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN make setup

EXPOSE ${MELLOW_PORT}

# Disable policy-rc.d to avoid service start issues
RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d

CMD ["python", "-u", "mellow_db/server_run.py"]
