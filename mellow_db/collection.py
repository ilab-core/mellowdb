import logging
import os
import random
import shutil
import time
from functools import wraps

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from mellow_db import metadata_db, vector_db
from mellow_db.exceptions import (CollectionExistsError,
                                  CollectionNotExistsError,
                                  MissingInfoInCollection)
from mellow_db.storage import (download_from_gcs, validate_folder,
                               validate_gcs_folder)
from mellow_db.utils import count_different

load_dotenv()
# The directory where the collection data will be stored
DATA_DIR = os.getenv("MELLOW_DATA_DIR", "./mellow_data/collection_data")


class Collection:
    """
    Represents a collection of vector and metadata databases.
    This class provides an interface for managing a collection that consists of a FAISS-based
    vector database and a SQLite-based metadata database.

    Attributes:
        name (str): The name of the collection.
        collection_dir (str): The directory path where the collection is stored.
        vector_db_ (FaissIndex): An instance of the FAISS vector database.
        meta_db_ (MetadataDB): An instance of the metadata database.
    """

    def __init__(self, name, index_config=None, schema=None, not_exists_ok=False):
        """
        Initializes the Collection instance.
        Checks if the collection exists (unless `not_exists_ok` is True), initializes the
        vector and metadata databases, and sets up database connections.

        Args:
            name (str): The name of the collection.
            index_config (dict, optional): Configuration settings for the FAISS index.
            schema (dict, optional): Schema definition for the metadata database.
            not_exists_ok (bool, optional): If False, raises an error if the collection does not exist.
                Defaults to False.

        Raises:
            CollectionNotExistsError: If `not_exists_ok` is False and the specified collection does not exist.
        """
        self.logger = logging.getLogger("collection")
        if not not_exists_ok and not Collection.exists(name):
            raise CollectionNotExistsError(
                f"Could not find the collection '{name}' in '{DATA_DIR}'")
        self.name = name
        self.collection_dir = Collection._get_collection_dir(self.name)
        vector_db_url, metadata_db_url = self._get_db_urls(self.collection_dir)
        self.vector_db_ = vector_db.FaissIndex(vector_db_url, index_config)
        self.meta_db_ = metadata_db.MetadataDB(metadata_db_url, schema)

    @classmethod
    def create(cls, name, schema, index_config=None):
        """
        Create a new collection in the specified directory with the given schema and optional index configuration.

        Args:
            name (str): The name of the collection to be created.
            schema (dict): The schema for the collection, defining the structure of the data.
            index_config (dict, optional): Configuration for the index, if applicable. Defaults to None.

        Returns:
            Collection: A new `Collection` instance initialized with the provided parameters.

        Raises:
            CollectionExistsError: If a collection with the same name already exists in the specified directory.
        """
        if Collection.exists(name):
            raise CollectionExistsError(f"Collection '{name}' already exists")
        os.makedirs(Collection._get_collection_dir(name), exist_ok=True)
        # TODO add here metadata of collection e.g. who created when
        return cls(name=name,
                   index_config=index_config,
                   schema=schema,
                   not_exists_ok=True)

    @staticmethod
    def load_handler(func):
        """
        Decorator that manages the creation and cleanup of a collection directory
        when loading a collection.
        This ensures that:
        - The collection directory is created before loading.
        - If an error occurs during loading, the directory is removed to prevent
          incomplete or corrupted collections.

        Args:
            func (callable): The function to be decorated.

        Returns:
            callable: The wrapped function with collection directory handling.
        """
        @wraps(func)
        def wrapper(cls, name, *args, **kwargs):
            if name is None:
                raise ValueError("Missing required argument: 'name'")

            if cls.exists(name):
                raise CollectionExistsError(f"Collection '{name}' already exists")

            collection_dir = cls._get_collection_dir(name)
            os.makedirs(collection_dir)
            try:
                return func(cls, name, *args, collection_dir=collection_dir, **kwargs)
            except Exception as e:
                shutil.rmtree(collection_dir)
                raise e
        return wrapper

    @classmethod
    @load_handler
    def load_from_path(cls, name, path, collection_dir=None):
        """
        Loads a collection from a local directory.
        This function copies the necessary database files from the specified
        directory (`path`) into a newly created collection directory.

        Args:
            name (str): The name of the collection.
            path (str): The local directory containing the collection data.
            collection_dir (str, optional): The target directory for the collection.
                Managed by the decorator.

        Returns:
            cls: An instance of the collection.
        """
        shutil.copy(os.path.join(path, "vector_db.db"), collection_dir)
        shutil.copy(os.path.join(path, "metadata_db.db"), collection_dir)
        return cls(name=name, index_config=None, schema=None, not_exists_ok=False)

    @classmethod
    @load_handler
    def load_from_gcs(cls, name, creds, bucket, gcs_path, collection_dir=None):
        """
        Loads a collection from Google Cloud Storage (GCS).
        This function downloads the necessary database files from a GCS bucket
        into a newly created collection directory.

        Args:
            name (str): The name of the collection.
            creds: Credentials for accessing GCS.
            bucket (str): The name of the GCS bucket.
            gcs_path (str): The GCS path where the collection data is stored.
            collection_dir (str, optional): The target directory for the collection.
                                            Managed by the decorator.

        Returns:
            cls: An instance of the collection.
        """
        download_from_gcs(creds, bucket, os.path.join(gcs_path, "vector_db.db"), collection_dir)
        download_from_gcs(creds, bucket, os.path.join(gcs_path, "metadata_db.db"), collection_dir)
        return cls(name=name, index_config=None, schema=None, not_exists_ok=False)

    def get_count(self):
        """
        Get the item count of the vector and metadata databases, ensuring that they are consistent.

        Returns:
            int: The item count of the vector database, assuming both the vector
                and metadata databases are of equal count.

        Raises:
            MissingInfoInCollection: If the count of the vector database does not match
                the count of the metadata database, this exception is raised with a message
                indicating the mismatch.
        """
        vector_count = self.vector_db_.get_count()
        meta_count = self.meta_db_.get_count()
        if vector_count != meta_count:
            raise MissingInfoInCollection(
                f"Vector item count {vector_count} and metadata item count {meta_count} are not matching. Check logs for debugging")
        return vector_count

    def get_info(self):
        meta_info = self.meta_db_.get_info()
        vector_info = self.vector_db_.get_info()
        info = {
            **meta_info,
            **vector_info,
            "size_in_bytes": vector_info["size_in_bytes"] + meta_info["size_in_bytes"]
        }
        return info

    def add(self, key_to_data=None, key_to_metadata=None, upsert=False):
        """
        Add new data and/or metadata to the collection, with optional upsert functionality.

        Args:
            key_to_data (dict, optional): A dictionary mapping keys to their corresponding data (e.g., embeddings).
                The keys serve as unique identifiers for the data being added to the vector database.
                Defaults to None, meaning no data will be added.
            key_to_metadata (dict, optional): A dictionary mapping keys to their corresponding metadata.
                The keys serve as unique identifiers for the metadata being added to the metadata database.
                Defaults to None, meaning no metadata will be added.
            upsert (bool, optional): Whether to update existing entries if the key already exists.
                Defaults to False, meaning duplicate keys will raise an error.

        Raises:
            Exception: If an error occurs during the operation, the transaction is rolled back for both the
                vector and metadata databases, and the original exception is re-raised.

        Returns:
            self: The current instance of the collection, allowing for method chaining.

        Behavior:
            - If `key_to_data` is provided, the data will be added to the vector database.
            - If `key_to_metadata` is provided, the metadata will be added to the metadata database.
            - If both are provided, both operations will be performed.
            - Both `key_to_data` and `key_to_metadata` being `None` should ideally be prohibited in client functions.
            - Transactions are committed only if all operations are successful.
            - In the event of an error, all changes are rolled back to maintain consistency.
        """
        try:
            # Handle embeddings if available
            if key_to_data:
                self.vector_db_.add(key_to_data, upsert=upsert)
            # Handle metadata if available
            if key_to_metadata:
                self.meta_db_.insert(key_to_metadata, upsert=upsert)
            # commit since there is no error above this line
            self.vector_db_.commit(allow_empty_commit=not bool(key_to_data))
            self.meta_db_.commit()
        except Exception as e:
            self.vector_db_.rollback()
            self.meta_db_.rollback()
            raise e
        return self

    def get(self, where={}, projection=["key"], limit=None):
        """
        Retrieve data from the collection by querying metadata and/or vector data.

        Args:
            where (dict, optional): A dictionary representing filtering conditions for the metadata query.
                Defaults to an empty dictionary, meaning no filtering.
            projection (list, optional): A list of fields to include in the results.
                Defaults to ["key"], which only includes the "key" field in the results.
            limit (int, optional): The maximum number of rows to return.
                If `where` is provided, the limit is applied after metadata filtering.
                Defaults to None, meaning no limit is applied.

        Returns:
            list: A list of tuples, where each tuple represents a result row.
                Each tuple contains the values for the fields specified in `projection`,
                preserving the order of `projection`.
        """
        returning_projection, vector_projection, meta_projection = Collection._get_projection(projection, ["key"])
        metadata_results = None
        if where or meta_projection:
            meta_projection = list(set(meta_projection) | {"key"})
            metadata_results = self.meta_db_.query(
                where=where, projection=meta_projection, limit=limit, return_field_names=True)
        if vector_projection:
            # retrieve only for subset_keys if metadata filtering is set
            # subset_keys=None means retrieving all items stored in vector_db_
            subset_keys = [meta["key"] for meta in metadata_results] if metadata_results is not None else None
            results = self.vector_db_.query(
                keys=subset_keys, projection=vector_projection, limit=limit, return_field_names=True)
        else:
            # iterate in metadata_results if vector results is not requested
            results = metadata_results
        # return without field names and in the same order with returning_projection
        return [tuple(row[field] if field in vector_projection else metadata_results[ix][field]
                      for field in returning_projection)
                for ix, row in enumerate(results)]

    def search(self, query_keys, where={}, projection=["key", "similarity"], n_results=20,
               threshold=None, not_exists_ok=False):
        """
        Perform a similarity search on vector data with optional filtering by metadata.

        Args:
            query_keys (list): A list of keys to query the vector database for similarity search.
            where (dict, optional): A dictionary representing filtering conditions for the metadata query.
                Defaults to an empty dictionary, meaning no filtering.
            projection (list, optional): A list of fields to include in the results.
                Defaults to ["key", "similarity"], where:
                    - "key": The identifier of the similar item.
                    - "similarity": The similarity score between the query and the result.
            n_results (int, optional): The maximum number of results to return for each query key.
                Defaults to 20.
            threshold (float, optional): Minimum similarity score required for results to be included.
                Defaults to None, meaning no threshold is applied.
            not_exists_ok (bool, optional): Whether to allow missing keys in the vector database without raising an error.
                Defaults to False, meaning an error will be raised if query keys are missing.

        Returns:
            list: A list of tuples, where each tuple contains tuples of
                (field_1, field_2, ..., field_n) for the top similar items based on `projection`,
                ordered as specified in `projection`.
                - Each tuple corresponds to a result for a query key.
                - If `not_exists_ok` is True and a query key is not found, the corresponding entry in the outer list is `None`.
                - If no results satisfy the conditions for a query key, the corresponding entry in the outer list is an empty tuple.
        """
        returning_projection, vector_projection, meta_projection = Collection._get_projection(
            projection, ["key", "similarity"], is_search=True)
        subset_keys = None
        # if there is metadata information requested
        if where or meta_projection:
            meta_projection = list(set(meta_projection) | {"key"})
            subset_metadata = self.meta_db_.query(where=where, projection=meta_projection, return_field_names=True)
            subset_metadata = {row["key"]: row for row in subset_metadata}
            if where:
                # embedding search within these ids
                subset_keys = list(subset_metadata.keys())

        results = self.vector_db_.search(
            query_keys=query_keys, subset_keys=subset_keys, n_results=n_results, threshold=threshold,
            projection=vector_projection, not_exists_ok=not_exists_ok, return_field_names=True)
        # return without field names and in the same order with returning_projection
        return [tuple(tuple(similar[field] if field in vector_projection else subset_metadata[similar["key"]][field]
                            for field in returning_projection)
                for similar in similars) if similars is not None else None
                for similars in results]

    def delete(self, not_exists_ok=False):
        """
        Delete the collection, including both the vector data and metadata, and remove the associated directory.

        Args:
            not_exists_ok (bool, optional): Whether to suppress errors when the collection does not exist.
                If `True`, no error will be raised if the collection is missing. Defaults to `False`, meaning
                an error will be raised if the collection cannot be found.

        Raises:
            CollectionNotExistsError: If the collection does not exist and `not_exists_ok` is `False`.

        Returns:
            self: The current instance of the collection, allowing for method chaining.

        Behavior:
            - Attempts to get the item count of the collection.
              If it is missing or corrupted, a warning is printed, and the item count is treated as 0.
            - Drops the vector data and metadata from their respective databases.
            - If the collection's directory exists, it is deleted.
            - If `not_exists_ok` is `True` or if the metadata_db or vector_db are missing (item count is 0),
              no error is raised.
            - If the collection is missing and `not_exists_ok` is `False`, a `CollectionNotExistsError` is raised.
            - Prints a success message when the collection is deleted.
        """
        try:
            item_count = self.get_count()
        except MissingInfoInCollection:
            self.logger.warning(self._make_log({"msg": "Corrupted info in collection"}))
            item_count = 0
        # if item_count 0, not_exists_ok=True
        self.vector_db_.drop(not_exists_ok=(not_exists_ok or not item_count))
        # if meta_count 0, not_exists_ok=True
        self.meta_db_.drop(not_exists_ok=(not_exists_ok or not item_count))
        if os.path.exists(self.collection_dir):
            shutil.rmtree(self.collection_dir)
        else:
            if not not_exists_ok:
                raise CollectionNotExistsError(f"Collection '{self.name}' does not exist")
        self.logger.info(self._make_log({"msg": "Successfully deleted"}))
        return self

    def eval(self, subset_size, test_size, k=30, tolerance=1e-5, batch_size=500):
        """
        Evaluate the performance of the similarity search using both mellow similarity and cosine similarity,
        and compute precision, recall, and differences between the two methods.

        Args:
            subset_size (int): The number of keys to sample from the vector database for the subset.
            test_size (int): The number of keys to sample from the vector database for testing.
            k (int, optional): The number of similar items to retrieve for each test key. Defaults to 30.
            tolerance (float, optional): The tolerance threshold for comparing similarity scores. Defaults to 1e-5.
            batch_size (int, optional): The batch size to use when processing the test keys. Defaults to 500.
                                        The `batch_size` is fixed in the server-side and is not controlled by the client.

        Returns:
            tuple: A tuple containing the following evaluation metrics:
                - avg_mellow_time (float): The average time taken to compute mellow similarities across all batches.
                - avg_cosine_time (float): The average time taken to compute cosine similarities across all batches.
                - avg_precision (float): The average precision score between mellow and cosine similarity for all test keys.
                - avg_recall (float): The average recall score between mellow and cosine similarity for all test keys.
                - sum_diffs (int): The total number of different scores between mellow and cosine similarity results across all test keys.
                - avg_diffs (float): The average number of different scores between mellow and cosine similarity results per test key.

        Behavior:
            - Samples `subset_size` keys and `test_size` keys from the vector database for evaluation.
            - For each batch of `test_size` keys:
                - Computes similarities using the mellow search method and measures the time taken.
                - Computes similarities using cosine similarity and measures the time taken.
            - Calculates precision, recall, and the number of different scores between the mellow and cosine similarity results.
            - Returns the average time, precision, recall, and differences across all test keys.
        """
        all_keys = self.vector_db_.get_keys()
        # limit the size of all embeddings because of cosine similarity cost
        subset_keys = random.sample(all_keys, subset_size)
        test_keys = random.sample(all_keys, test_size)

        mellow_results = {}
        cosine_results = {}
        mellow_times = []
        cosine_times = []
        for i in range(0, len(test_keys), batch_size):
            batch_keys = test_keys[i:i + batch_size]
            start_time = time.time()
            mellow_similars = self.search(query_keys=batch_keys,
                                          where={"key": {"$in": subset_keys}},
                                          projection=["key", "similarity"],
                                          n_results=k)
            mellow_times.append(time.time() - start_time)
            mellow_results.update({
                key: {mellow_similars[ix][j][0]: mellow_similars[ix][j][1] for j in range(k)}
                for ix, key in enumerate(batch_keys)
            })
            start_time = time.time()
            query_embeddings = np.array(self.vector_db_.get_embeddings(batch_keys, reshape=None))
            subset_embeddings = np.array(self.vector_db_.get_embeddings(subset_keys, reshape=None))
            cosine_similars = cosine_similarity(query_embeddings, subset_embeddings)
            cosine_times.append(time.time() - start_time)
            cosine_results.update({
                key: {subset_keys[j]: float(cosine_similars[ix][j]) for j in np.argsort(-cosine_similars[ix])[:k]}
                for ix, key in enumerate(batch_keys)
            })

        precisions = [
            precision_score(list(cosine_results[test_key].keys()),
                            list(mellow_results[test_key].keys()),
                            average='macro',
                            zero_division=0)
            for test_key in test_keys
        ]
        recalls = [
            recall_score(list(cosine_results[test_key].keys()),
                         list(mellow_results[test_key].keys()),
                         average='macro',
                         zero_division=0)
            for test_key in test_keys
        ]
        diffs = [
            count_different(list(cosine_results[test_key].values()),
                            list(mellow_results[test_key].values()),
                            tolerance=tolerance)
            for test_key in test_keys
        ]

        avg_mellow_time = sum(mellow_times) / len(mellow_times)
        avg_cosine_time = sum(cosine_times) / len(cosine_times)
        avg_precision = round(sum(precisions) / len(precisions), 2)
        avg_recall = round(sum(recalls) / len(recalls), 2)
        sum_diffs = sum(diffs)
        avg_diffs = round(sum_diffs / len(diffs), 2)

        return avg_mellow_time, avg_cosine_time, avg_precision, avg_recall, sum_diffs, avg_diffs

    def back_up(self, backup_dir, backup_name):
        """
        Back up the collection to a specified local directory.

        This method creates a backup of the collection by saving the necessary files to the specified directory
        on the local filesystem. If the directory does not exist, it will be created.

        Args:
            backup_dir (str): The path to the parent directory where the backup will be stored.
                This should be a valid local directory path. If the folder does not exist, it will be created.
            backup_name (str): The name of the backup folder or file to create within the `backup_dir`.
                It will be appended to the `backup_dir` to form the full path.
                This can be considered as a backup tag.

        Returns:
            backup_folder_path (str): The full path to the backup folder where the collection was saved.

        Raises:
            FileExistsError: If the backup destination exists but not empty.
            NotADirectoryError: If the provided backup directory is a file, not a directory.
        """
        backup_folder_path = os.path.join(backup_dir, backup_name)
        validate_folder(backup_folder_path)  # Ensures the folder exists and is valid

        # Perform the actual backup of the vector and metadata dbs
        self.vector_db_.back_up(backup_folder_path)
        self.meta_db_.back_up(backup_folder_path)

        return backup_folder_path

    def back_up_to_gcs(self, creds, bucket, backup_dir, backup_name):
        """
        Back up the collection to a specified path in Google Cloud Storage (GCS).

        This method creates a backup of the collection by uploading the necessary files to the specified directory
        in the GCS bucket. If the directory does not exist, it will be created.

        Args:
            creds (str): Google Cloud service account credentials JSON string. This is used for
                authenticating the connection to Google Cloud Storage.
            bucket (str): The name of the Google Cloud Storage bucket where the backup will be stored.
            backup_dir (str): The GCS path to the parent directory where the backup will be stored.
                This should be a valid path within the specified GCS bucket.
                If the folder does not exist, it will be created.
            backup_name (str): The name of the backup folder or file to create within the `backup_dir`.
                It will be appended to the `backup_dir` to form the full path.
                This can be considered as a backup tag.

        Returns:
            backup_folder_path: The full path to the backup folder where the collection was saved in GCS.

        Raises:
            google.api_core.exceptions.NotFound: If the specified GCS folder or bucket is not found.
            ValueError: If the provided path is not a valid GCS folder path or if the folder is not empty.
        """
        backup_folder_path = os.path.join(backup_dir, backup_name)
        validate_gcs_folder(creds, bucket, backup_folder_path)  # Ensures the folder in GCS is valid or can be created

        # Perform the actual backup of the vector and meta databases to GCS
        self.vector_db_.back_up_to_gcs(creds, bucket, backup_folder_path)
        self.meta_db_.back_up_to_gcs(creds, bucket, backup_folder_path)

        return backup_folder_path

    @staticmethod
    def exists(name):
        """
        Check if a collection with the specified name exists in the given data directory.

        Args:
            name (str): The name of the collection to check.

        Returns:
            bool: `True` if the collection exists, `False` otherwise.

        Note:
            - This method does not check if the collection is properly initialized;
                it only checks for the existence of the directory.
        """
        collection_dir = Collection._get_collection_dir(name)
        return os.path.exists(collection_dir)

    @staticmethod
    def _get_db_urls(collection_dir):
        """
        Generate database URLs for the vector and metadata databases.

        Args:
            collection_dir (str): The directory path of the collection.

        Returns:
            tuple: A tuple containing:
                - vector_db_url (str): URL for the FAISS vector database.
                - metadata_db_url (str): URL for the SQLite metadata database.
        """
        vector_db_path, metadata_db_path = Collection._get_db_paths(collection_dir)
        return f"faiss:///{vector_db_path}", f"sqlite:///{metadata_db_path}"

    @staticmethod
    def _get_db_paths(collection_dir):
        """
        Get file paths for the vector and metadata databases.

        Args:
            collection_dir (str): The directory path of the collection.

        Returns:
            tuple: A tuple containing:
                - vector_db_path (str): Path to the FAISS vector database file.
                - metadata_db_path (str): Path to the SQLite metadata database file.
        """
        vector_db_path = os.path.join(collection_dir, "vector_db.db")
        metadata_db_path = os.path.join(collection_dir, "metadata_db.db")
        return vector_db_path, metadata_db_path

    @staticmethod
    def _get_collection_dir(name):
        """
        Get the directory path for a given collection name.

        Args:
            name (str): The name of the collection.

        Returns:
            str: The full path to the collection's directory.
        """
        return os.path.join(DATA_DIR, name)

    @staticmethod
    def _get_projection(projection, default_list, vector_db_columns={"key", "embedding", "similarity"},
                        is_search=False):
        if not projection:
            projection = default_list
        projection_set = set(projection)
        including = projection_set & vector_db_columns
        excluding = projection_set - vector_db_columns
        # ensure 'key' is included for joining results if both projection lists have values
        # or if it is a search with metadata filtering (in that case we need 'key' to join results)
        if (including and excluding) or (is_search and excluding):
            including = including | {"key"}
            excluding = excluding | {"key"}
        return projection, list(including), list(excluding)

    def _make_log(self, extra={}):
        msg = {
            "collection": self.name,
            "extra": extra
        }
        return msg
