import json
import os
import pickle
import shutil

import pandas as pd
import pytest
from dotenv import load_dotenv

from mellow_db.client import MellowClient
from mellow_db.collection import Collection
from mellow_db.exceptions import CollectionNotExistsError
from mellow_db.metadata_db import MetadataDB
from mellow_db.storage import gcs_bucket_connection
from mellow_db.vector_db import FaissIndex

load_dotenv(override=True)

gcp_creds_ = json.loads(os.getenv("GCP_SERVICE_ACCOUNT"))
metadata_db = None
client_ = MellowClient(os.getenv("MELLOW_HOST"),
                       os.getenv("MELLOW_PORT"),
                       gcp_creds_)
collection_name = "pytest_collection_in_server"
dir_ = "tests/pytest_data/"
os.makedirs(dir_, exist_ok=True)
os.makedirs(f"{dir_}empty/", exist_ok=True)
gcs_bucket_ = "mellow_db"

collection_schema = [("PublishDate", "integer", False),
                     ('Publisher', "string", True),
                     ('Lang', "string", False),
                     ('BookTitle', "string", False)]
client_collection_schema = [
    {"field_name": "PublishDate", "field_type": "integer", "is_nullable": False},
    {"field_name": "Publisher", "field_type": "string", "is_nullable": True},
    {"field_name": "Lang", "field_type": "string", "is_nullable": False},
    {"field_name": "BookTitle", "field_type": "string", "is_nullable": False}]

df_operator_map = {
    "$eq": lambda col, val: col == val,  # Equal
    "$ne": lambda col, val: col != val if val is not None else col.notna(),  # Not equal, handling None
    "$gt": lambda col, val: col > val,  # Greater than
    "$lt": lambda col, val: col < val,  # Less than
    "$gte": lambda col, val: col >= val,  # Greater than or equal
    "$lte": lambda col, val: col <= val,  # Less than or equal
    "$in": lambda col, val: col.isin(val) if None not in val else col.isin(val) | col.isna(),  # In list, handling None
}


@pytest.fixture(scope="module")
def gcp_creds():
    return gcp_creds_


@pytest.fixture(scope="module")
def gcs_bucket():
    return gcs_bucket_


@pytest.fixture(scope="module")
def base_dir():
    return dir_


@pytest.fixture(scope="module")
def embeddings():
    emb = {}
    with open("tests/mock_data/embeddings/type1.pkl", 'rb') as file:
        emb["type_1"] = pickle.load(file)

    with open("tests/mock_data/embeddings/type2.pkl", 'rb') as file:
        emb["type_2"] = pickle.load(file)

    with open("tests/mock_data/embeddings/type3.pkl", 'rb') as file:
        emb["type_3"] = pickle.load(file)

    return emb


@pytest.fixture(scope="module")
def metadata():
    meta = {}
    with open("tests/mock_data/metadata/type3.pkl", 'rb') as file:
        meta["type_3"] = pickle.load(file)

    return meta


@pytest.fixture(scope="module")
def metadata_df(metadata):
    metadata = [{**{"key": k}, **val} for k, val in metadata["type_3"].items()]
    return pd.DataFrame(metadata)


@pytest.fixture(scope="module")
def faiss_urls_with_data():
    urls = {}
    urls["type_1"] = f"faiss:///{dir_}/vector_db.db"
    urls["type_2"] = f"faiss:///{dir_}/type2_vectors.db"
    return urls


@pytest.fixture()
def faiss_urls():  # empty urls
    urls = {}
    urls["type_1"] = f"faiss:///{dir_}/empty/vector_db.db"
    FaissIndex(urls["type_1"]).drop(not_exists_ok=True)
    urls["type_2"] = f"faiss:///{dir_}/empty/type2_vectors.db"
    FaissIndex(urls["type_2"]).drop(not_exists_ok=True)
    return urls


@pytest.fixture(scope="module")
def indexes(embeddings, faiss_urls_with_data):
    indexes_with_data = {}
    # indexes can be loaded from gcs but creating again for testing made more sense
    indexes_with_data["type_1"] = FaissIndex(
        faiss_urls_with_data["type_1"]).drop(not_exists_ok=True).add(embeddings["type_1"]).commit()
    indexes_with_data["type_2"] = FaissIndex(
        faiss_urls_with_data["type_2"]).drop(not_exists_ok=True).add(embeddings["type_2"]).commit()
    return indexes_with_data


@pytest.fixture(scope="module")
def collection_(embeddings, metadata, base_dir):
    collection = Collection.create("mock_collection",
                                   index_config={"index_type": "Flat", "index_metric": "METRIC_INNER_PRODUCT"},
                                   schema=collection_schema)
    collection.add(key_to_data=embeddings["type_3"], key_to_metadata=metadata["type_3"], upsert=False)
    yield collection
    collection.delete()


@pytest.fixture()
def db():
    global metadata_db
    if metadata_db is not None:
        metadata_db.drop(not_exists_ok=True)
        metadata_db.close()
    metadata_db = MetadataDB(db_url=f"sqlite:///{dir_}/metadata_db.db",
                             schema=[("name", "string", False),
                                     ("age", "integer", False),
                                     ("hobbies", "list", True),
                                     ("info", "string", True)])
    return metadata_db


@pytest.fixture()
def client(gcp_creds):
    client = MellowClient(os.getenv("MELLOW_HOST"),
                          os.getenv("MELLOW_PORT"),
                          gcp_creds)
    client.create_collection(collection_name="temp_pytest_collection", collection_schema=client_collection_schema)
    client.use_collection("temp_pytest_collection")
    return client


@pytest.fixture(scope="module")
def client_with_data(embeddings, metadata):
    client_.create_collection(collection_name=collection_name, collection_schema=client_collection_schema)
    client_.use_collection(collection_name)
    client_.add(key_to_data=embeddings["type_3"], key_to_metadata=metadata["type_3"], upsert=False)
    return client_


def delete_collection(name):
    MellowClient(os.getenv("MELLOW_HOST"),
                 os.getenv("MELLOW_PORT"),
                 gcp_creds_).delete_collection(name)


def apply_where_conditions(df, where):
    condition = None
    for col, filters in where.items():
        for operator, value in filters.items():
            new_condition = df_operator_map[operator](df[col], value)
            if condition is None:
                condition = new_condition
            else:
                condition &= new_condition
    return df[condition]


def pytest_sessionfinish(session, exitstatus):
    print("Clearing temporary pytest data")
    client_.disconnect()
    try:
        delete_collection(collection_name)
    except CollectionNotExistsError:
        pass
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    if os.path.exists(f"{dir_}empty/"):
        shutil.rmtree(f"{dir_}empty/")

    bucket = gcs_bucket_connection(gcp_creds_, gcs_bucket_)
    blobs = bucket.list_blobs(prefix=dir_)
    for blob in blobs:
        blob.delete()
