import os
import shutil

import pytest
import sqlalchemy

from mellow_db.exceptions import ResourceNotFoundError
from mellow_db.metadata_db import MetadataDB
from mellow_db.storage import gcs_bucket_connection


def test_insert_and_read(db):
    db.insert({"1": {"name": "Alice", "age": 25, "hobbies": ["reading", "swimming"]}}).commit()
    record = db.read("1")
    assert record.name == "Alice"
    assert record.age == 25
    assert record.hobbies == ["reading", "swimming"]


def test_upsert(db):
    db.insert({"1": {"name": "Alice", "age": 25, "hobbies": ["reading"]}}).commit()
    db.insert({"1": {"name": "Alice", "age": 30, "hobbies": ["reading", "traveling"]}}, upsert=True).commit()
    record = db.read("1")
    assert record.age == 30
    assert record.hobbies == ["reading", "traveling"]


def test_insert_duplicate_key(db):
    db.insert({"1": {"name": "Alice", "age": 25}}).commit()
    with pytest.raises(ValueError):
        db.insert({"1": {"name": "Bob", "age": 30}}).commit()


def test_update(db):
    db.insert({"1": {"name": "Alice", "age": 25}}).commit()
    db.update("1", {"age": 30, "name": "Alicia"}).commit()
    record = db.read("1")
    assert record.age == 30
    assert record.name == "Alicia"


@pytest.mark.parametrize(
    "filter, projection, return_fields, expected",
    [
        ({"age": {"$gte": 30}}, ["name"], False, [("Bob",), ("Charlie",)]),
        ({"age": {"$lt": 30}}, ["name"], False, [("Alice",)]),
        ({"name": {"$eq": "Alice"}}, ["name"], False, [("Alice",)]),
        ({"name": {"$ne": "Alice"}}, ["name"], False, [("Bob",), ("Charlie",)]),
        ({"age": {"$in": [25, 35]}}, ["name"], False, [("Alice",), ("Charlie",)]),
        ({"age": {"$gt": 35}}, ["name"], False, []),
        # return_fields = True
        ({"age": {"$in": [10, 70]}}, ["name"], True, []),
        ({"name": {"$eq": "Alicia"}}, ["name"], True, []),
        ({"age": {"$gte": 30}}, ["name", "age"], True,
         [{"name": "Bob", "age": 30}, {"name": "Charlie", "age": 35}]),
        ({"age": {"$gte": 30}}, ["age", "name", "key"], True,
         [{"age": 30, "name": "Bob", "key": "2"}, {"age": 35, "name": "Charlie", "key": "3"}]),
        ({"age": {"$lt": 30}}, ["key"], True, [{"key": "1"}]),
        # projection tests
        ({"age": {"$gte": 30}}, ["name", "age"], False, [("Bob", 30), ("Charlie", 35)]),
        ({"age": {"$gte": 30}}, ["age", "name", "key"], False, [(30, "Bob", "2"), (35, "Charlie", "3")]),
        ({"age": {"$lt": 30}}, ["key"], False, [('1',)]),
    ]
)
def test_query(db, filter, projection, return_fields, expected):
    db.insert({
        "1": {"name": "Alice", "age": 25},
        "2": {"name": "Bob", "age": 30},
        "3": {"name": "Charlie", "age": 35},
    }).commit()
    results = db.query(where=filter, projection=projection, return_field_names=return_fields)
    assert results == expected


@pytest.mark.parametrize(
    "filter, expected",
    [
        ({"info": {"$eq": None}}, [{"key": "1"}, {"key": "2"}]),
        ({"info": {"$ne": None}}, [{"key": "3"}, {"key": "4"}, {"key": "5"}]),
        ({"info": {"$in": ["abc"]}}, [{"key": "5"}]),
        ({"info": {"$in": [None]}}, [{"key": "1"}, {"key": "2"}]),
        ({"info": {"$in": ["abc", None]}}, [{"key": "1"}, {"key": "2"}, {"key": "5"}]),
    ]
)
def test_query_with_null(db, filter, expected):
    db.insert({
        "1": {"name": "Dumbledore", "age": 105, "info": None},
        "2": {"name": "Harry", "age": 17, "info": None},
        "3": {"name": "Sirius", "age": 35, "info": "sad emoji"},
        "4": {"name": "Dobby", "age": 21, "info": "freedom"},
        "5": {"name": "Hermione", "age": 17, "info": "abc"},
    }).commit()
    results = db.query(where=filter, return_field_names=True)
    assert results == expected


@pytest.mark.parametrize(
    "filter, projection, exception",
    [
        ({"age": {"$unsupported": 30}}, ["key"], ValueError),
        ({"age": {"$eq": 25}}, [], sqlalchemy.exc.InvalidRequestError),
        ({"age": {"$eq": 25}}, ["abc"], AttributeError),
        ({"abc": {"$eq": 25}}, ["key"], AttributeError),
    ]
)
def test_query_unsupported(db, filter, projection, exception):
    db.insert({"1": {"name": "Alice", "age": 25}}).commit()
    with pytest.raises(exception):
        db.query(filter, projection)


def test_delete(db):
    db.insert({"1": {"name": "Alice", "age": 25}}).commit()
    db.delete("1").commit()
    record = db.read("1")
    assert record is None


def test_drop(db):
    db.drop(not_exists_ok=True)
    with pytest.raises(ResourceNotFoundError):
        db.drop(not_exists_ok=False)


@pytest.mark.parametrize(
    "schema, exception",
    [
        ([("key", "string", False)], ValueError),
        ([("similarity", "string", True)], ValueError),
        ([("embedding", "float", False)], ValueError),
        ([("similarity", "string", True), ("embedding", "integer", False)], ValueError),
        ([("field1", "string", True), ("field2", "integer", True)], ValueError),
        ([("field1", "string", True), ("field2", "float", True)], ValueError),
    ]
)
def test_validate_schema(db, schema, exception):
    with pytest.raises(exception):
        db._validate_schema(schema)


def test_get_info(db):
    info_ = db.get_info()
    assert isinstance(info_['size_in_bytes'], int)
    del info_['size_in_bytes']
    assert info_ == {
        'primary_keys': ['key'],
        'meta_columns': [
            {'name': 'key', 'type': 'VARCHAR', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'name', 'type': 'VARCHAR', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'age', 'type': 'INTEGER', 'is_nullable': False, 'is_index': False, 'default': None},
            {'name': 'hobbies', 'type': 'JSON', 'is_nullable': True, 'is_index': False, 'default': None},
            {'name': 'info', 'type': 'VARCHAR', 'is_nullable': True, 'is_index': False, 'default': None}
        ],
        'item_count': 0
    }


# this will not run periodically with ci/cd pipelines
def test_back_up_and_load(db, base_dir):
    # back up
    backup_dir = os.path.join(base_dir, "temp_metadata_db_backup")
    db.back_up(backup_dir)
    assert os.path.exists(os.path.join(backup_dir, "metadata_db.db"))

    # load from backup
    loaded_db = MetadataDB.load_from_path(
        os.path.join(backup_dir, "metadata_db.db"),
        db_url=f"sqlite:///{base_dir}/metadata_db.db",
        schema=[("name", "string", False),
                ("age", "integer", False),
                ("hobbies", "list", True),
                ("info", "string", True)])
    assert isinstance(loaded_db, MetadataDB)
    shutil.rmtree(backup_dir)

    # test error cases
    with open(backup_dir, "w") as f:
        f.write("dummy text")
    with pytest.raises(NotADirectoryError):
        db.back_up(backup_dir)
    os.remove(backup_dir)
    with pytest.raises(FileNotFoundError):
        MetadataDB.load_from_path(
            os.path.join(backup_dir, "metadata_db.db"),
            db_url=f"sqlite:///{base_dir}/metadata_db.db",
            schema=[("name", "string", False),
                    ("age", "integer", False),
                    ("hobbies", "list", True),
                    ("info", "string", True)])


def test_back_up_to_gcs_and_load_from_gcs(gcp_creds, gcs_bucket, base_dir, db):
    # back up
    bucket = gcs_bucket_connection(gcp_creds, gcs_bucket)
    folder_path = os.path.join(base_dir, "temp_metadata_db_backup")
    path = os.path.join(folder_path, "metadata_db.db")
    blob = bucket.blob(path)
    assert not blob.exists()
    db.back_up_to_gcs(gcp_creds, gcs_bucket, folder_path)
    assert blob.exists()

    # load from backup
    loaded_db = db.load_from_gcs(
        gcp_creds, gcs_bucket, path,
        db_url=f"sqlite:///{base_dir}/metadata_db.db",
        schema=[("name", "string", False),
                ("age", "integer", False),
                ("hobbies", "list", True),
                ("info", "string", True)])
    assert isinstance(loaded_db, MetadataDB)
    blob.delete()

    # test error cases
    dummy_blob = bucket.blob(folder_path)
    dummy_blob.upload_from_string("dummy text")
    with pytest.raises(ValueError):
        db.back_up_to_gcs(gcp_creds, gcs_bucket, folder_path)
    dummy_blob.delete()
    with pytest.raises(ResourceNotFoundError):
        MetadataDB.load_from_gcs(
            gcp_creds, gcs_bucket, path,
            db_url=f"sqlite:///{base_dir}/metadata_db.db",
            schema=[("name", "string", False),
                    ("age", "integer", False),
                    ("hobbies", "list", True),
                    ("info", "string", True)])
