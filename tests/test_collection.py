import os
import shutil

import numpy as np
import pytest

from mellow_db.collection import Collection
from mellow_db.exceptions import (CollectionExistsError,
                                  CollectionNotExistsError,
                                  MissingInfoInCollection,
                                  ResourceNotFoundError)
from mellow_db.storage import gcs_bucket_connection
from tests.conftest import collection_schema


def test_create(base_dir):
    col = Collection.create("temp_collection", schema=collection_schema)
    assert Collection.exists("temp_collection")
    with pytest.raises(CollectionExistsError):
        col = Collection.create("temp_collection", schema=[])
    col.delete(not_exists_ok=False)  # restore


def test_add(base_dir, embeddings, metadata):
    col = Collection.create("temp_collection", schema=collection_schema)
    col.add(key_to_data=embeddings["type_3"], key_to_metadata=metadata["type_3"], upsert=False)
    assert col.get_count() == len(embeddings["type_3"])
    with pytest.raises(ValueError):
        col.add(key_to_data=embeddings["type_3"], key_to_metadata=metadata["type_3"], upsert=False)
    col.add(key_to_data=embeddings["type_3"], key_to_metadata=metadata["type_3"], upsert=True)
    assert col.get_count() == len(embeddings["type_3"])
    col.delete(not_exists_ok=False)  # restore


@pytest.mark.parametrize("where, projection, limit, check_idx, expected_values, expected_count", [
    ({"BookTitle": {"$eq": "Book MRF"}}, ["key", "BookTitle", "embedding"],
     None, 1, ["Book MRF"], 262),
    ({"BookTitle": {"$in": ["Book QMV", "Book VIH"]}},
     ["Publisher", "BookTitle", "PublishDate"], None, 0, [None], 2),
    ({}, ["Lang"], None, 0, ["tr", "en", "de"], 500),
    ({}, None, None, 0, None, 500),
    ({"Lang": {"$ne": "de"}}, ["Lang"], None, 0, ["tr", "en"], 499),
    ({"Publisher": {"$eq": "Publisher GHGZG"}}, ["Publisher"],
     None, 0, ["Publisher GHGZG"], 1),
    ({"Publisher": {"$eq": None}}, ["Publisher", "embedding"], None, 0, [None], 174),
    ({"Publisher": {"$in": ["Publisher GHGZG"]}}, ["Publisher", "embedding"],
     None, 0, ["Publisher GHGZG"], 1),
    ({"Publisher": {"$in": [None]}}, ["Publisher"], None, 0, [None], 174),
    ({"Publisher": {"$in": ["Publisher GHGZG", None]}}, ["Publisher"],
     None, 0, ["Publisher GHGZG", None], 175),
    ({"Publisher": {"$in": [None, "Publisher GHGZG"]}}, ["Publisher"],
     None, 0, ["Publisher GHGZG", None], 175),
    ({"Publisher": {"$ne": "Publisher GHGZG"}}, ["Publisher"], None, 0, None, 499),
    ({"Publisher": {"$ne": None}}, ["Publisher", "embedding"], None, 0, None, 326),
    # limit tests
    ({"Publisher": {"$in": ["Publisher GHGZG", None]}}, ["Publisher"],
     1880, 0, ["Publisher GHGZG", None], 175),
    ({"Publisher": {"$in": ["Publisher GHGZG", None]}}, ["Publisher"],
     102, 0, ["Publisher GHGZG", None], 102),
    ({"Publisher": {"$ne": None}}, ["Publisher", "embedding"], 10, 0, None, 10),
    ({"Publisher": {"$ne": None}}, ["Publisher", "embedding"], 1, 0, None, 1),
    ({}, None, 100, 0, None, 100),
])
def test_get(collection_, where, projection, limit, check_idx, expected_values, expected_count):
    result = collection_.get(where=where, projection=projection, limit=limit)
    assert all(len(row) == (len(projection) if projection else 1) for row in result)
    assert expected_values is None or all(row[check_idx] in expected_values for row in result)
    assert len(result) == expected_count
    if projection and "embedding" in projection:
        emb_index = projection.index("embedding")
        assert all(isinstance(row[emb_index], list) and np.array(row[emb_index]).shape == (1024,) for row in result)


@pytest.mark.parametrize(
    "projection, default_list, vector_columns, is_search, expected_projection, expected_including, expected_excluding",
    [
        (["field1", "key", "field2"], ["key", "field2"], {"field2"}, False,
         ["field1", "key", "field2"], {"field2", "key"}, {"field1", "key"}),
        (["field1", "field2"], ["key", "field4"], {"field3"}, False,
         ["field1", "field2"], set(), {"field1", "field2"}),
        ([], ["key", "field1"], {"key", "field1"}, False,
         ["key", "field1"], {"key", "field1"}, set()),
        (["field4", "field5"], ["key", "field1"], {"key", "field1"}, False,
         ["field4", "field5"], set(), {"field4", "field5"}),
        (["field4", "field5"], ["key", "field1"], {"key", "field1"}, True,
         ["field4", "field5"], {"key"}, {"key", "field4", "field5"}),
    ]
)
def test_get_projection(projection, default_list, vector_columns, is_search,
                        expected_projection, expected_including, expected_excluding):
    res = Collection._get_projection(projection, default_list, vector_columns, is_search=is_search)
    assert res[0] == expected_projection
    assert set(res[1]) == expected_including
    assert set(res[2]) == expected_excluding


@pytest.mark.parametrize("query_keys, where, projection, n_results, threshold, not_exists_ok, expected", [
    (
        ["1451"], {}, ["key"], 12, None, False,
        [(('1451',), ('1029',), ('1274',), ('1415',), ('1008',), ('1151',), ('1314',), ('1011',),
         ('1260',), ('1429',), ('1312',), ('1310',))]
    ),
    (
        ["1451", "abc"], {}, ["key"], 12, None, True,
        [(('1451',), ('1029',), ('1274',), ('1415',), ('1008',), ('1151',),
         ('1314',), ('1011',), ('1260',), ('1429',), ('1312',), ('1310',)), None]
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
        ["1453", "1452"], {}, ["PublishDate", "similarity", "BookTitle", "key", "Lang"], 4, None, False,
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
        ["1453", "xyz", "abc", "1452", "def", "qqq"], {},
        ["PublishDate", "similarity", "BookTitle", "key", "Lang"], 4, None, True,
        [((1725638160, 1.0, 'Book UIY', '1453', 'tr'),
          (1726668600, 0.64, 'Book LRN', '1243', 'tr'),
          (1725974400, 0.64, 'Book JTC', '1223', 'tr'),
          (1724229900, 0.63, 'Book IYA', '1213', 'tr')),
         None,
         None,
         ((1725621060, 1.0, 'Book MRF', '1452', 'tr'),
          (1725299580, 0.88, 'Book MRF', '1130', 'tr'),
          (1725355800, 0.87, 'Book MRF', '1142', 'tr'),
          (1725298980, 0.84, 'Book MRF', '1096', 'tr')),
         None,
         None]
    ),
    (
        ["1228"], {"BookTitle": {"$in": ["Book END"]}, "Lang": {"$eq": "en"}},
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
def test_search(collection_, query_keys, where, projection, n_results, threshold, not_exists_ok, expected):
    if isinstance(expected, type):  # if an expection
        with pytest.raises(expected):
            res = collection_.search(query_keys, where, projection, n_results, threshold, not_exists_ok)
    else:
        res = collection_.search(query_keys, where, projection, n_results, threshold, not_exists_ok)
        if "similarity" in projection:
            sim_index = projection.index("similarity")
            res = [tuple(tuple(row_item[i] if i != sim_index else round(row_item[i], 2)
                               for i in range(len(projection)))
                         for row_item in row)
                   if row is not None else None
                   for row in res]
        assert res == expected


def test_get_count(collection_):
    assert collection_.get_count() == 500


def test_missing_info_in_collection(embeddings):
    col = Collection.create("temp_collection", schema=collection_schema)
    col.add(key_to_data=embeddings["type_3"], upsert=False)
    with pytest.raises(MissingInfoInCollection):
        col.get_count()
    col.delete(not_exists_ok=False)  # restore


def test_delete():
    col = Collection.create("temp_collection", schema=[])
    col.delete(not_exists_ok=False)
    assert col.get_count() == 0
    col.delete(not_exists_ok=True)
    assert col.get_count() == 0
    with pytest.raises(CollectionNotExistsError):
        col.delete(not_exists_ok=False)


def test_get_info(collection_):
    info_ = collection_.get_info()
    assert isinstance(info_['size_in_bytes'], int)
    del info_['size_in_bytes']
    assert info_ == {
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
def test_back_up_and_load(collection_, base_dir):
    # back up
    backup_dir = os.path.join(base_dir, "temp_collection_backup")
    collection_.back_up(base_dir, "temp_collection_backup")
    assert os.path.exists(os.path.join(backup_dir, "vector_db.db"))
    assert os.path.exists(os.path.join(backup_dir, "metadata_db.db"))

    # load from backup
    loaded_collection1 = Collection.load_from_path("collection_from_backup1", backup_dir)
    assert isinstance(loaded_collection1, Collection)

    # test error cases
    with pytest.raises(FileExistsError):
        collection_.back_up(base_dir, "temp_collection_backup")
    shutil.rmtree(backup_dir)
    with open(backup_dir, "w") as f:
        f.write("dummy text")
    with pytest.raises(NotADirectoryError):
        collection_.back_up(base_dir, "temp_collection_backup")
    os.remove(backup_dir)
    with pytest.raises(FileNotFoundError):
        Collection.load_from_path("collection_from_backup2", backup_dir)

    # restore
    loaded_collection1.delete()


def test_back_up_to_gcs(collection_, gcp_creds, gcs_bucket, base_dir):
    # back up
    bucket = gcs_bucket_connection(gcp_creds, gcs_bucket)
    folder_path = os.path.join(base_dir, "temp_collection_backup")
    vector_blob = bucket.blob(os.path.join(base_dir, "temp_collection_backup", "vector_db.db"))
    metadata_blob = bucket.blob(os.path.join(base_dir, "temp_collection_backup", "metadata_db.db"))
    assert not vector_blob.exists()
    assert not metadata_blob.exists()
    collection_.back_up_to_gcs(gcp_creds, gcs_bucket, base_dir, "temp_collection_backup")
    assert vector_blob.exists()
    assert metadata_blob.exists()

    # load from backup
    loaded_collection1 = Collection.load_from_gcs(
        "collection_from_backup1", gcp_creds, gcs_bucket, folder_path)
    assert isinstance(loaded_collection1, Collection)

    # test error cases
    with pytest.raises(ValueError):
        collection_.back_up_to_gcs(gcp_creds, gcs_bucket, base_dir, "temp_collection_backup")
    vector_blob.delete()
    metadata_blob.delete()
    dummy_blob = bucket.blob(folder_path)
    dummy_blob.upload_from_string("dummy text")
    with pytest.raises(ValueError):
        collection_.back_up_to_gcs(gcp_creds, gcs_bucket, base_dir, "temp_collection_backup")
    dummy_blob.delete()
    with pytest.raises(ResourceNotFoundError):
        Collection.load_from_gcs("collection_from_backup2", gcp_creds, gcs_bucket, folder_path)

    # restore
    loaded_collection1.delete()
