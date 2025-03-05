from collections.abc import Iterable, Mapping

import grpc
import yaml
from google.cloud import secretmanager


def load_yaml(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML file as a dictionary.
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def access_secret(client, project_id, secret_id):
    """
    Retrieve the latest version of a secret from Google Cloud Secret Manager.

    Args:
        client (secretmanager.SecretManagerServiceClient): A Secret Manager client.
        project_id (str): Google Cloud project ID.
        secret_id (str): The ID of the secret.

    Returns:
        str: The secret value.
    """
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    secret_value = response.payload.data.decode("UTF-8")
    return secret_value


def load_creds(service_account_info, host):
    """
    Load SSL/TLS credentials from Google Cloud Secret Manager.

    Args:
        service_account_info (dict): GCS service account credentials as a dictionary.
        host (str): The target host ('client' or 'server').

    Returns:
        tuple[bytes, bytes, bytes]: The certificate, private key, and root certificate.
    """
    secret_client = secretmanager.SecretManagerServiceClient.from_service_account_info(service_account_info)
    project_id = service_account_info.get("project_id")
    crt_ = access_secret(secret_client, project_id, f"sm_{host}_crt").encode()
    key_ = access_secret(secret_client, project_id, f"sm_{host}_key").encode()
    root_crt = access_secret(secret_client, project_id, "sm_ca_crt").encode()
    return crt_, key_, root_crt


def get_client_credentials(service_account_info):
    """
    Get gRPC client SSL/TLS credentials.

    Args:
        service_account_info (dict): GCS service account credentials to access the SSL/TLS credentials.

    Returns:
        grpc.ChannelCredentials: The gRPC client SSL credentials.
    """
    crt_, key_, root_crt = load_creds(service_account_info, "client")
    return grpc.ssl_channel_credentials(
        root_certificates=root_crt,
        private_key=key_,
        certificate_chain=crt_
    )


def get_server_credentials(service_account_info):
    """
    Get gRPC server SSL/TLS credentials.

    Args:
        service_account_info (dict): GCS service account credentials to access the SSL/TLS credentials.

    Returns:
        grpc.ServerCredentials: The gRPC server SSL credentials.
    """
    crt_, key_, root_crt = load_creds(service_account_info, "server")
    return grpc.ssl_server_credentials(
        [(key_, crt_)],
        require_client_auth=True,
        root_certificates=root_crt
    )


def is_list_compatible(val):
    """
    Check if a value is an iterable (excluding strings, bytes, and mappings).

    Args:
        val (any): The value to check.

    Returns:
        bool: True if `val` is an iterable and not a string, bytes, or mapping.
    """
    return isinstance(val, Iterable) and not isinstance(val, (str, bytes, Mapping))


def count_different(true_scores, predicted_scores, tolerance=1e-6):
    """
    Count the number of elements that differ beyond a specified tolerance.

    Args:
        true_scores (iterable of float): The ground truth values.
        predicted_scores (iterable of float): The predicted values.
        tolerance (float, optional): The tolerance for considering values different.
            Defaults to 1e-6.

    Returns:
        int: The number of differing elements.
    """
    return sum(abs(a - b) > tolerance for a, b in zip(true_scores, predicted_scores))
