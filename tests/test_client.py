import os
import threading

import pandas as pd
import pytest

from mellow_db.client import MellowClient
from mellow_db.exceptions import (ActiveConnectionFound, CollectionExistsError,
                                  CollectionHasActiveClientsError,
                                  CollectionNotExistsError,
                                  ConcurrentWriteError, ConnectionFailedError,
                                  InvalidRequest)
from mellow_db.storage import gcs_bucket_connection
from tests.conftest import (apply_where_conditions, collection_name,
                            delete_collection)


def test_add_get_collection_count(client, embeddings, metadata):
    emb = embeddings["type_3"]
    meta = metadata["type_3"]
    keys = list(emb.keys())
    emb_batch1 = {k: emb[k] for k in keys[:200]}
    meta_batch1 = {k: meta[k] for k in keys[:200]}
    count = client.add(key_to_data=emb_batch1, key_to_metadata=meta_batch1, upsert=False)
    assert count == 200
    emb_batch2 = {k: emb[k] for k in keys[100:280]}
    meta_batch2 = {k: meta[k] for k in keys[100:280]}
    with pytest.raises(ValueError):
        client.add(key_to_data=emb_batch2, key_to_metadata=meta_batch2, upsert=False)
    assert client.get_collection_item_count() == 200
    count = client.add(key_to_data=emb_batch2, key_to_metadata=meta_batch2, upsert=True)
    assert count == 280

    # restore
    client.disconnect()
    delete_collection("temp_pytest_collection")


@pytest.mark.parametrize("where, projection, expected", [
    (
        {"key": {"$eq": "1002"}}, ["key", "BookTitle", "PublishDate", "Publisher", "Lang"],
        [('1002', 'Book XXH', 1724162100.0, 'Publisher QZUSV', 'tr')]
    ),
    (
        {"Publisher": {"$in": ["Publisher QZUSV",
                               "Publisher AKJFO"]}},
        ["key", "BookTitle"],
        [('1002', 'Book XXH'), ('1059', 'Book XDH'),
         ('1302', 'Book END'), ('1435', 'Book CLQ')]
    ),
    ({"BookTitle": {"$in": [None]}}, ["PublishDate", "Lang", "BookTitle", "embedding", "key"], []),
    ({"BookTitle": {"$in": ["abc"]}}, ["PublishDate", "Lang", "BookTitle", "embedding", "key"], []),
    ({"BookTitle": {"$eq": "abc"}}, ["PublishDate", "Lang", "BookTitle", "embedding", "key"], []),
    ({"key": {"$eq": "1002"}, "BookTitle": {"$in": ["Book MRF", "Book VGL"]}},
     ["key", "BookTitle", "PublishDate", "Publisher", "Lang", "embedding"], []),
    (
        {"PublishDate": {"$gte": 1727709240.0}}, ["key"],
        [("1058",), ("1217",), ("1308",), ("1358",), ("1445",)]
    ),
    (
        {"PublishDate": {"$gt": 1727709240.0}}, ["key"], [("1058",), ("1308",), ("1358",), ("1445",)]
    ),
    (
        {"PublishDate": {"$gt": 1727709240.0}, "Publisher": {"$ne": None}}, ["Lang", "key", "BookTitle"],
        [("tr", "1308", "Book END")]
    ),
])
def test_get(client_with_data, where, projection, expected):
    assert client_with_data.get(where, projection) == expected


@pytest.mark.parametrize("where, projection, limit, expected_count", [
    ({"key": {"$in": ["1003"]}}, ["key", "Publisher"], 5, 1),
    ({"Publisher": {"$in": ["Publisher GHGZG", None]}}, ["Publisher"], 1880, 175),
    ({"Publisher": {"$in": ["Publisher GHGZG", None]}}, ["Publisher"], 102, 102),
    ({"Publisher": {"$ne": None}}, ["Publisher", "embedding"], 10, 10),
    ({"Publisher": {"$ne": None}}, ["Publisher", "embedding"], 1, 1),
    ({}, None, 100, 100),
])
def test_get_with_limit(client_with_data, where, projection, limit, expected_count):
    assert len(client_with_data.get(where, projection, limit)) == expected_count


@pytest.mark.parametrize("where, projection", [
    (
        {"Lang": {"$in": ["tr"]}, "BookTitle": {"$in": ["Book MRF", "Book XXH"]}},
        ["key", "BookTitle", "Lang", "embedding"]
    ),
    (
        {"key": {"$in": ["1001", "1159", "1500", "1330", "1343", "1342", "1341", "1162"]}},
        ["BookTitle", "embedding", "Lang"]
    ),
    (
        {"BookTitle": {"$ne": "Book MRF"}, "PublishDate": {
            "$lt": 1727709240.0}, "Publisher": {"$ne": "Education / Training"}},
        ["embedding", "Lang"]
    ),
])
def test_get_auto_check(client_with_data, where, projection, embeddings, metadata_df):
    meta_projection = [p for p in projection if p != "embedding"]
    result = client_with_data.get(where, projection)
    result = pd.DataFrame(result, columns=projection).sort_values(by=meta_projection)
    filtered_metadata = apply_where_conditions(metadata_df, where).sort_values(by=meta_projection)
    pd.testing.assert_frame_equal(
        result[meta_projection].reset_index(drop=True),
        filtered_metadata[meta_projection].reset_index(drop=True)
    )
    if "embedding" in projection:
        vector_projection = [p for p in projection if p in {"embedding", "key"}]
        expected_embeddings = [(k, embeddings["type_3"][k]) for k in filtered_metadata["key"].values.tolist()]
        expected_embeddings = pd.DataFrame(result, columns=("key", "embedding"))
        pd.testing.assert_frame_equal(
            result[vector_projection].reset_index(drop=True),
            expected_embeddings[vector_projection].reset_index(drop=True)
        )


@pytest.mark.parametrize("query_keys, where, projection, n_results, threshold, not_exists_ok, expected", [
    (
        ["1451"], None, ["key"], 12, None, False,
        [(('1451',), ('1029',), ('1274',), ('1415',), ('1008',), ('1151',),
         ('1314',), ('1011',), ('1260',), ('1429',), ('1312',), ('1310',))]
    ),
    (
        ["1451", "abc"], None, ["key"], 12, None, True,
        [(('1451',), ('1029',), ('1274',), ('1415',), ('1008',), ('1151',),
         ('1314',), ('1011',), ('1260',), ('1429',), ('1312',), ('1310',)), None]
    ),
    (
        ["1453", "1452"], None, ["key", "similarity", "BookTitle"], 3, None, False,
        [(('1453', 1.0, 'Book UIY'),
          ('1243', 0.64, 'Book LRN'),
          ('1223', 0.64, 'Book JTC')),
         (('1452', 1.0, 'Book MRF'),
          ('1130', 0.88, 'Book MRF'),
          ('1142', 0.87, 'Book MRF'))]
    ),
    (
        ["1453", "1452"], None, ["key", "similarity", "BookTitle"], 3, 0.1, False,
        [(('1453', 1.0, 'Book UIY'),
          ('1243', 0.64, 'Book LRN'),
          ('1223', 0.64, 'Book JTC')),
         (('1452', 1.0, 'Book MRF'),
          ('1130', 0.88, 'Book MRF'),
          ('1142', 0.87, 'Book MRF'))]
    ),
    (
        ["1453", "abc", "1452"], {}, ["key", "similarity", "BookTitle"], 3, None, True,
        [(('1453', 1.0, 'Book UIY'),
          ('1243', 0.64, 'Book LRN'),
          ('1223', 0.64, 'Book JTC')),
         None,
         (('1452', 1.0, 'Book MRF'),
          ('1130', 0.88, 'Book MRF'),
          ('1142', 0.87, 'Book MRF'))]
    ),
    (
        ["1453", "1452"], None, ["PublishDate", "similarity", "BookTitle", "key", "Lang"], 4, None, False,
        [((1725638160, 1.0, 'Book UIY', '1453', 'tr'),
          (1726668600, 0.64, 'Book LRN', '1243', 'tr'),
          (1725974400, 0.64, 'Book JTC', '1223', 'tr'),
          (1724229900, 0.63, 'Book IYA', '1213', 'tr')),
         ((1725621060, 1.0, 'Book MRF', '1452', 'tr'),
          (1725299580, 0.88, 'Book MRF', '1130', 'tr'),
          (1725355800, 0.87, 'Book MRF', '1142', 'tr'),
          (1725298980, 0.84, 'Book MRF', '1096', 'tr'))]
    ),
    (
        ["1453"], {"BookTitle": {"$in": ["Book END"]}, "Lang": {"$eq": "en"}},
        ["key", "similarity", "BookTitle", "Lang"], 4, None, False,
        [()]
    ),
    (
        ["1453"], {"BookTitle": {"$in": ["Book END"]}, "Lang": {"$eq": "en"}},
        ["key", "similarity", "BookTitle", "Lang"], 70, 0.1, False,
        [()]
    ),
    (
        ["1453", "1452"], {"BookTitle": {"$in": ["Book END"]}},
        ["key", "similarity", "BookTitle", "Lang"], 4, 0.8, False,
        [(), ()]
    ),
    (
        ["1453"], {"BookTitle": {"$in": ["Book ZAO"]}},
        ["key", "similarity", "BookTitle", "Lang"], 4, None, True,
        [(('1228', 0.53, 'Book ZAO', 'tr'),)]
    ),
    (
        ["xyz", "1453", "1452", "abc"], {"BookTitle": {"$in": ["Book END"]}},
        ["key", "similarity", "BookTitle", "Lang"], 4, 0.8, True,
        [None, (), (), None]
    ),
    (
        ["1453", "abc", "1452"], {}, ["key", "similarity", "BookTitle"], 3, None, False, KeyError
    ),
    (
        ["1453", "abc", "1452"], {}, ["key", "similarity", "field3"], 3, None, False, AttributeError
    ),
])
def test_search(client_with_data, query_keys, where, projection, n_results, threshold, not_exists_ok, expected):
    if isinstance(expected, type):  # if an expection
        with pytest.raises(expected):
            res = client_with_data.search(query_keys, where, projection, n_results, threshold, not_exists_ok)
    else:
        res = client_with_data.search(query_keys, where, projection, n_results, threshold, not_exists_ok)
        if "similarity" in projection:
            sim_index = projection.index("similarity")
            res = [tuple(tuple(row_item[i] if i != sim_index else round(row_item[i], 2)
                               for i in range(len(projection)))
                         for row_item in row)
                   if row is not None else None
                   for row in res]
        assert res == expected


def test_create_collection(gcp_creds):
    client = MellowClient(os.getenv("MELLOW_HOST"),
                          os.getenv("MELLOW_PORT"),
                          gcp_creds)
    client.create_collection("temp_pytest_collection", collection_schema=[])
    client.use_collection("temp_pytest_collection")
    assert client.get_collection_item_count() == 0

    with pytest.raises(CollectionExistsError):
        client.create_collection("temp_pytest_collection", collection_schema=[])

    client.create_collection("temp_pytest_collection_2", collection_schema=[])

    # restore
    client.disconnect()
    delete_collection("temp_pytest_collection")
    delete_collection("temp_pytest_collection_2")


def test_use_collection(gcp_creds):
    client = MellowClient(os.getenv("MELLOW_HOST"),
                          os.getenv("MELLOW_PORT"),
                          gcp_creds)
    client.create_collection("temp_pytest_collection", collection_schema=[])
    with pytest.raises(InvalidRequest):
        client.get_collection_item_count()
    client.use_collection("temp_pytest_collection")
    client.get_collection_item_count()  # valid request
    with pytest.raises(ActiveConnectionFound):
        client.use_collection("temp_pytest_collection")
    with pytest.raises(ActiveConnectionFound):
        client.use_collection("temp_pytest_collection_2")

    # restore
    client.disconnect()
    delete_collection("temp_pytest_collection")


def test_delete_collection(client, gcp_creds):
    client.disconnect()
    client2 = MellowClient(os.getenv("MELLOW_HOST"),
                           os.getenv("MELLOW_PORT"),
                           gcp_creds)
    client2.delete_collection("temp_pytest_collection")
    with pytest.raises(CollectionNotExistsError):
        client2.use_collection("temp_pytest_collection")
    client2.use_collection(collection_name)  # valid collection
    with pytest.raises(CollectionHasActiveClientsError):
        client2.delete_collection(collection_name)

    # restore
    client2.disconnect()


def test_disconnect(client):
    client.disconnect()
    with pytest.raises(ConnectionFailedError):
        client.search(["3916551"])
    with pytest.raises(ConnectionFailedError):
        client.disconnect()

    delete_collection("temp_pytest_collection")


def test_get_collection_info(client_with_data):
    info_ = client_with_data.get_collection_info()
    assert isinstance(info_['size_in_bytes'], int)
    del info_['size_in_bytes']
    assert info_ == {
        'name': 'pytest_collection_in_server',
        'primary_keys': ['key'],
        'meta_columns': [
            {'name': 'key', 'type': 'VARCHAR', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'PublishDate', 'type': 'INTEGER', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'Publisher', 'type': 'VARCHAR', 'is_nullable': True, 'is_index': False, 'default': None},
            {'name': 'Lang', 'type': 'VARCHAR', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'BookTitle', 'type': 'VARCHAR', 'is_nullable': False, 'is_index': False, 'default': None}
        ],
        'item_count': 500,
        'faiss_index_type': 'Flat',
        'faiss_index_metric': 'METRIC_INNER_PRODUCT',
        'embedding_dim': 1024
    }


# this will not run periodically with ci/cd pipelines
def test_back_up_and_load(client, embeddings, metadata, gcp_creds, base_dir):
    # setup
    keys = list(embeddings["type_3"].keys())
    emb_batch1 = {k: embeddings["type_3"][k] for k in keys[:20]}
    meta_batch1 = {k: metadata["type_3"][k] for k in keys[:20]}
    cols = list(meta_batch1[keys[0]].keys()) + ["embedding"]
    client.add(key_to_data=emb_batch1, key_to_metadata=meta_batch1)

    # back up
    _, saved_backup_path = client.back_up(base_dir)
    assert os.path.exists(saved_backup_path)

    # load from backup
    client.load_from_path(saved_backup_path, "collection_from_backup")
    with pytest.raises(CollectionExistsError):
        client.create_collection("collection_from_backup", collection_schema=[])
    with pytest.raises(CollectionExistsError):
        client.load_from_path(saved_backup_path, "collection_from_backup")

    # check equality
    check_client = MellowClient(
        os.getenv("MELLOW_HOST"), os.getenv("MELLOW_PORT"), gcp_creds)
    check_client.use_collection("collection_from_backup")
    assert client.get(where={}, projection=cols) == check_client.get(where={}, projection=cols)

    # should not raise an error bc it saves with timestamp
    _, saved_backup_path2 = client.back_up(base_dir)
    assert os.path.exists(saved_backup_path2)
    assert saved_backup_path2 != saved_backup_path

    # restore
    client.disconnect()
    check_client.disconnect()
    delete_collection("temp_pytest_collection")
    delete_collection("collection_from_backup")


def test_back_up_to_gcs(client, embeddings, metadata, gcp_creds, gcs_bucket, base_dir):
    # setup
    keys = list(embeddings["type_3"].keys())
    emb_batch1 = {k: embeddings["type_3"][k] for k in keys[:20]}
    meta_batch1 = {k: metadata["type_3"][k] for k in keys[:20]}
    cols = list(meta_batch1[keys[0]].keys()) + ["embedding"]
    client.add(key_to_data=emb_batch1, key_to_metadata=meta_batch1)

    # back up
    _, saved_backup_path = client.back_up_to_gcs(gcp_creds, gcs_bucket, base_dir)
    bucket = gcs_bucket_connection(gcp_creds, gcs_bucket)
    vector_db_blob = bucket.blob(os.path.join(saved_backup_path, "vector_db.db"))
    meta_db_blob = bucket.blob(os.path.join(saved_backup_path, "metadata_db.db"))
    assert vector_db_blob.exists()
    assert meta_db_blob.exists()

    # load from backup
    client.load_from_gcs(
        gcp_creds, gcs_bucket, saved_backup_path, "collection_from_backup")
    with pytest.raises(CollectionExistsError):
        client.create_collection("collection_from_backup", collection_schema=[])
    with pytest.raises(CollectionExistsError):
        client.load_from_gcs(
            gcp_creds, gcs_bucket, saved_backup_path, "collection_from_backup")

    # check equality
    check_client = MellowClient(
        os.getenv("MELLOW_HOST"), os.getenv("MELLOW_PORT"), gcp_creds)
    check_client.use_collection("collection_from_backup")
    assert client.get(where={}, projection=cols) == check_client.get(where={}, projection=cols)

    # should not raise an error bc it saves with timestamp
    _, saved_backup_path2 = client.back_up_to_gcs(gcp_creds, gcs_bucket, base_dir)
    assert saved_backup_path2 != saved_backup_path
    vector_db_blob = bucket.blob(os.path.join(saved_backup_path2, "vector_db.db"))
    meta_db_blob = bucket.blob(os.path.join(saved_backup_path2, "metadata_db.db"))
    assert vector_db_blob.exists()
    assert meta_db_blob.exists()

    # restore
    client.disconnect()
    check_client.disconnect()
    delete_collection("temp_pytest_collection")
    delete_collection("collection_from_backup")


# this will not run periodically with ci/cd pipelines
def test_concurrent_operations(client, embeddings, metadata):
    emb = embeddings["type_3"]
    meta = metadata["type_3"]

    keys = list(emb.keys())
    emb = {k: emb[k] for k in keys[:200]}
    meta = {k: meta[k] for k in keys[:200]}

    client.add(key_to_data=emb, key_to_metadata=meta, upsert=True)
    add_results, get_results, search_results, threads = [], [], [], []

    def adder():
        for i in range(10):
            try:
                result = client.add(
                    key_to_data=emb,
                    key_to_metadata=meta,
                    upsert=True)
                add_results.append(result)
            except ConcurrentWriteError:
                add_results.append(200)

    def getter():
        for i in range(10):
            result = client.get(
                {},
                projection=["BookTitle", "embedding", "Lang", "key", "PublishDate"])
            get_results.append(result)

    def searcher():
        for i in range(10):
            result = client.search(keys[i:i + 3], n_results=20, not_exists_ok=False)
            search_results.append(result)

    # Create threads
    for _ in range(5):
        threads.append(threading.Thread(target=adder))
        threads.append(threading.Thread(target=getter))
        threads.append(threading.Thread(target=searcher))

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Assert all operations were successful
    assert all(result == 200 for result in add_results)
    assert all(len(result) == 200 for result in get_results)
    assert all(len(result) == 3 for result in search_results)

    # restore
    client.disconnect()
    delete_collection("temp_pytest_collection")
