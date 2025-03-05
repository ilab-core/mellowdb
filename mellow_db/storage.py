import os
import pickle

import google
from google.cloud import storage

from mellow_db.exceptions import ResourceNotFoundError


def validate_folder(folder_path, not_empty_ok=False):
    """
    Validate and create a local folder if it does not exist.

    Args:
        folder_path (str): The path to the folder to be validated.
        not_empty_ok (bool, optional): If False, raises an error if the folder exists but is not empty.
            Defaults to False.

    Raises:
        NotADirectoryError: If a file exists at the given folder path.
        FileExistsError: If a non-empty folder exists and `not_empty_ok` is False.
    """
    if os.path.exists(folder_path):
        # Check if the folder_path is a "file"
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(
                f"Expected an empty directory at {folder_path}, but found a file with the same path.")

        # Check if the folder_path is a non-empty directory
        elif os.listdir(folder_path) and not not_empty_ok:
            raise FileExistsError(
                f"Expected an empty directory at {folder_path}, but found a non-empty directory.")
    else:
        os.makedirs(folder_path)


def validate_gcs_folder(creds_info, bucket_name, folder_path, not_empty_ok=False):
    """
    Validate a folder in a Google Cloud Storage (GCS) bucket.

    Args:
        creds_info (dict): Service account credentials for GCS authentication.
        bucket_name (str): The name of the GCS bucket.
        folder_path (str): The folder path in the GCS bucket.
        not_empty_ok (bool, optional): If False, raises an error if the folder exists but is not empty.
            Defaults to False.

    Raises:
        ValueError: If a file exists at the specified folder path.
        ValueError: If a non-empty folder exists and `not_empty_ok` is False.
    """
    bucket = gcs_bucket_connection(creds_info, bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_path))
    if blobs:
        # if the folder_path is a "file"
        if len(blobs) == 1 and blobs[0].name == folder_path:
            raise ValueError(f"Expected an empty directory at '{folder_path}', but found a file.")

        if not not_empty_ok:
            # if the folder_path is a non-empty directory
            raise ValueError(f"Expected an empty directory at '{folder_path}', but found a non-empty directory.")


def gcs_bucket_connection(creds_info, bucket_name):
    """
    Establish a connection to a Google Cloud Storage (GCS) bucket.

    Args:
        creds_info (dict): Service account credentials for GCS authentication.
        bucket_name (str): The name of the GCS bucket.

    Returns:
        google.cloud.storage.Bucket: A reference to the GCS bucket.
    """
    creds = google.oauth2.service_account.Credentials.from_service_account_info(creds_info)
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.get_bucket(bucket_name)
    return bucket


def load_pickle_from_gcs(creds_info, bucket_name, file_path):
    """
    Load a pickled file from Google Cloud Storage (GCS).

    Args:
        creds_info (dict): Service account credentials for GCS authentication.
        bucket_name (str): The name of the GCS bucket.
        file_path (str): The path to the pickle file in the GCS bucket.

    Returns:
        object: The unpickled Python object.

    Raises:
        ResourceNotFoundError: If the file does not exist in the specified location.
    """
    bucket = gcs_bucket_connection(creds_info, bucket_name)
    blob = bucket.blob(file_path)
    if blob.exists():
        with blob.open(mode='rb') as f:
            content = pickle.load(f)
        return content
    else:
        raise ResourceNotFoundError(f"File not found at '{file_path}'")


def download_from_gcs(creds_info, bucket_name, source_file_path, local_folder_path):
    """
    Download a file from Google Cloud Storage (GCS) to a local directory.

    Args:
        creds_info (dict): Service account credentials for GCS authentication.
        bucket_name (str): The name of the GCS bucket.
        source_file_path (str): The path to the file in the GCS bucket.
        local_folder_path (str): The local directory where the file should be downloaded.

    Returns:
        str: The local file path of the downloaded file.

    Raises:
        ResourceNotFoundError: If the file does not exist in the GCS bucket.
    """
    bucket = gcs_bucket_connection(creds_info, bucket_name)
    blob = bucket.blob(source_file_path)
    file_name = os.path.basename(source_file_path)
    local_file_path = os.path.join(local_folder_path, file_name)
    if blob.exists():
        blob.download_to_filename(local_file_path)
        return local_file_path
    else:
        raise ResourceNotFoundError(f"File not found at '{source_file_path}'")


def upload_to_gcs(creds_info, local_file_path, bucket_name, destination_folder_path):
    """
    Upload a local file to a specified folder in Google Cloud Storage (GCS).

    Args:
        creds_info (dict): Service account credentials for GCS authentication.
        local_file_path (str): The path to the local file to be uploaded.
        bucket_name (str): The name of the GCS bucket.
        destination_folder_path (str): The destination folder path in the GCS bucket.

    Returns:
        str: The full destination file path in the GCS bucket.
    """
    bucket = gcs_bucket_connection(creds_info, bucket_name)
    file_name = os.path.basename(local_file_path)
    destination_file_path = os.path.join(destination_folder_path, file_name)
    blob = bucket.blob(destination_file_path)
    blob.upload_from_filename(local_file_path)
    return destination_file_path
