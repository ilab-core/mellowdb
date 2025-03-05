import copy
import logging
import os
import pickle
import shutil
from itertools import islice

import faiss
import numpy as np

from mellow_db.decorators import with_rollback
from mellow_db.exceptions import ResourceNotFoundError
from mellow_db.storage import (download_from_gcs, upload_to_gcs,
                               validate_folder, validate_gcs_folder)
from mellow_db.utils import is_list_compatible


class FaissIndex():
    """
    A class for managing a Faiss-based vector index.

    Attributes:
        db_url (str): The URL pointing to the Faiss index storage location.
        db_path (str): The file path derived from `db_url` where the index is stored.
        index_type (str): The type of Faiss index used (default: "Flat").
        index_metric (str): The distance metric used in the index (default: "METRIC_INNER_PRODUCT").
    """

    def __init__(self, db_url, config=None):
        """
        Initialize a FaissIndex instance.

        If an existing index file is found, it loads the index; otherwise, it creates
        a new index with default or provided configurations.

        Args:
            db_url (str): The URL of the Faiss index storage.
            config (dict, optional): Configuration dictionary containing "index_type" and "index_metric".
                If not provided, defaults to "Flat" index with "METRIC_INNER_PRODUCT".
        """
        self.logger = logging.getLogger("faiss_index")
        self.db_url = db_url
        self.db_path = self.db_url.replace("faiss:///", "")
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.deserialize(data)
            # default stack variables
            self._clear_stack()
        else:
            self.logger.info(self._make_log({
                "msg": "Index does not exist. Creating new vector index"}
            ))
            if not config:
                self.index_type = "Flat"
                self.index_metric = "METRIC_INNER_PRODUCT"
            else:
                self.index_type = config["index_type"]
                self.index_metric = config["index_metric"]

            self.logger.info(self._make_log({
                "index_type": self.index_type,
                "index_metric": self.index_metric}
            ))
            # default attributes
            self._reset_data()

        self.logger.info(self._make_log({
            "msg": "FaissIndex ready", "num_items": self.get_count()}
        ))

    @classmethod
    def load_from_data(cls, data, db_url, upsert=False):
        """
        Load a Faiss index from a data dict and store it at the specified location.

        Args:
            data (dict): The data to be added to the index.
            db_url (str): The URL where the Faiss index should be stored.
            upsert (bool, optional): Whether to update existing entries if they already exist. Defaults to False.

        Returns:
            FaissIndex: An instance of FaissIndex with the loaded data.
        """
        instance = cls(db_url=db_url)
        instance.add(data, upsert=upsert).commit()

        instance.logger.info(instance._make_log({
            "msg": "FaissIndex successfully loaded", "num_items": instance.get_count()}
        ))
        return instance

    @classmethod
    def load_from_path(cls, file_path, db_url):
        """
        Load a Faiss index from a local file.

        Args:
            file_path (str): The path to the file containing the serialized Faiss index.
            db_url (str): The URL where the Faiss index should be stored.

        Returns:
            FaissIndex: An instance of FaissIndex with the loaded index.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        instance = cls(db_url=db_url)
        instance.deserialize(data)

        instance.logger.info(instance._make_log({
            "msg": "FaissIndex successfully loaded",
            "num_items": instance.get_count(),
            "data_path": file_path}
        ))
        return instance

    @classmethod
    def load_from_gcs(cls, creds, bucket, file_path, db_url):
        """
        Load a Faiss index from Google Cloud Storage (GCS).

        Args:
            creds (dict): Google Cloud credentials required for authentication.
            bucket (str): The name of the GCS bucket.
            file_path (str): The path to the index file in GCS.
            db_url (str): The URL where the Faiss index should be stored.

        Returns:
            FaissIndex: An instance of FaissIndex with the loaded index.
        """
        destination_folder = os.path.dirname(db_url.replace("faiss:///", ""))
        download_from_gcs(creds, bucket, file_path, destination_folder)
        instance = cls(db_url=db_url)

        instance.logger.info(instance._make_log({
            "msg": "FaissIndex successfully loaded",
            "num_items": instance.get_count(),
            "data_path": f'gs:/{bucket}/{file_path}'}
        ))
        return instance

    def get_count(self):
        """
        Get the number of items currently stored in the Faiss index.

        Returns:
            int: The number of indexed items.
        """
        return len(self.key_to_data)

    def get_file_size(self):
        """
        Get the file size of the Faiss index storage.

        Returns:
            int: The file size in bytes.
        """
        return os.path.getsize(self.db_path)

    def get_info(self):
        """
        Retrieve metadata information about the Faiss index.

        Returns:
            dict: A dictionary containing the following keys:
                - "faiss_index_type": The type of Faiss index.
                - "faiss_index_metric": The distance metric used.
                - "embedding_dim": The dimensionality of stored embeddings (-1 if empty).
                - "item_count": The number of stored embeddings.
                - "size_in_bytes": The file size of the index storage.
        """
        info = {
            "faiss_index_type": self.index_type,
            "faiss_index_metric": self.index_metric,
            "embedding_dim": next(iter(self.key_to_data.values())).shape[0] if self.key_to_data else -1,
            "item_count": self.get_count(),
            "size_in_bytes": self.get_file_size(),
        }
        return info

    def get_keys(self):
        """
        Retrieve all keys stored in the Faiss index.

        Returns:
            list: A list of keys representing stored embeddings.
        """
        return list(self.key_to_data.keys())

    def key_exists(self, key):
        """
        Check if a given key exists in the Faiss index.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.key_to_data

    def get_embeddings(self, keys=None, reshape=(1, -1)):
        """
        Retrieve embeddings associated with specified keys.

        Args:
            keys (list, optional): A list of keys for which embeddings are required.
                If None, returns embeddings for all stored keys.
            reshape (tuple, optional): Shape to which embeddings should be reshaped.
                Defaults to (1, -1).

        Returns:
            list: A list of NumPy arrays representing the requested embeddings.
        """
        if keys is None:
            return [v.reshape(reshape) for v in self.key_to_data.values()] if reshape else list(self.key_to_data.values())
        keys = FaissIndex._format_keys(keys)
        return [self.get_embedding(key=key, reshape=reshape) for key in keys]

    def get_embedding(self, key, reshape=(1, -1)):
        """
        Retrieve a single embedding associated with a given key.

        Args:
            key (str): The key for the embedding.
            reshape (tuple, optional): Shape to which the embedding should be reshaped.
                Defaults to (1, -1).

        Returns:
            np.ndarray: The embedding corresponding to the given key.
        """
        emb = self.key_to_data[key]
        return emb.reshape(reshape) if reshape else emb

    @with_rollback
    def add(self, key_to_data, upsert=False):
        """
        Add embeddings to the Faiss index temporarily. Changes are committed using `commit()`.

        Args:
            key_to_data (dict): A dictionary mapping keys to their corresponding embeddings.
            upsert (bool, optional): If True, updates existing keys with new embeddings.
                Defaults to False.

        Raises:
            ValueError: If `upsert` is False and any provided key already exists.

        Returns:
            FaissIndex: The updated FaissIndex instance.
        """
        if not upsert:
            existing_keys = [key_ for key_ in key_to_data if self.key_exists(key=key_)]
            if existing_keys:
                raise ValueError(
                    f"Key must be unique. {len(existing_keys)} embeddings already exists")
        self._stack_data_change(key_to_data)
        return self

    def query(self, keys=None, projection=["key", "embedding"], limit=None, return_field_names=False):
        """
        Query data for a set of keys, applying projections and optional row limits.

        Args:
            keys (list, optional): A list of keys to query. If None, queries all keys.
            projection (list): List of fields to include in the result. Supported fields: ["key", "embedding"].
            limit (int, optional): Maximum number of rows to return. If None, no limit is applied.
                This is only applied if keys is None.
            return_field_names (bool): If True, returns results as a list of dictionaries with field names as keys.
                If False, returns results as tuples.

        Returns:
            list: A list of dictionaries or tuples representing the queried data,
                depending on the value of `return_field_names`.
        """
        if keys is None:
            # Apply the limit only if keys is None
            if limit is not None:
                keys = list(islice(self.key_to_data.keys(), limit))
            else:
                # Using keys = self.get_keys() would be a better practice,
                # but self.key_to_data is more memory efficient, since keys is only used to iterate in
                keys = self.key_to_data
        else:
            keys = FaissIndex._format_keys(keys)

        projector = {
            "key": lambda key: key,
            "embedding": lambda key: self.get_embedding(key=key, reshape=None).tolist()
        }
        if return_field_names:
            results_fn = lambda key: {field: projector[field](key) for field in projection}
        else:
            results_fn = lambda key: tuple(projector[field](key) for field in projection)

        return [results_fn(key) for key in keys]

    def search(self, query_keys, subset_keys=None, n_results=20, threshold=None,
               projection=["key", "similarity"], not_exists_ok=False, return_field_names=False):
        """
        Perform a similarity search for the given query keys in the Faiss index.

        Args:
            query_keys (list or str): List of keys (or a single key) whose embeddings will be used for searching.
            subset_keys (list, optional): If provided, restricts the search space to only these keys.
            n_results (int, optional): Maximum number of similar items to retrieve per query. Defaults to 20.
            threshold (float, optional): Minimum similarity score required to include a result.
                Defaults to None, meaning all results are returned.
            projection (list, optional): Fields to include in the search results. Defaults to ["key", "similarity"].
                Available options: "key", "similarity", "embedding".
            not_exists_ok (bool, optional): If False, raises an error if a query key is missing. Defaults to False.
            return_field_names (bool, optional): If True, returns results as dictionaries with field names.
                Otherwise, returns tuples. Defaults to False.

        Returns:
            list: A list containing search results for each query key.
                  - If a query key does not exist and `not_exists_ok` is False, raises KeyError.
                  - If a query key does not exist and `not_exists_ok` is True, returns None for that key.
                  - Each valid query key returns a list of up to `n_results` similar items.
                    Returns empty list for the key, if no results are found for that key satisfying the criteria.
                    - If `return_field_names` is True, results are dictionaries
                        (e.g., {"key": "123", "similarity": 0.85}).
                    - Otherwise, results are tuples (e.g., ("123", 0.85)).
                  - If `threshold` is set, results below the threshold are filtered out.

        Raises:
            KeyError: If a query key is missing and `not_exists_ok` is False.

        Example:
            >>> index.search(["key1", "key2"], n_results=5, threshold=0.7)
            [[("123", 0.9), ("456", 0.8)], [("789", 0.85), ("234", 0.75)]]
        """
        # make query keys and embeddings
        query_keys = FaissIndex._format_keys(query_keys)
        found_keys = [key for key in query_keys if self.key_exists(key)]

        # not exists check
        if not not_exists_ok and len(query_keys) != len(found_keys):
            one_missing_key = None
            for key in query_keys:
                if key not in set(found_keys):
                    one_missing_key = key
                    break
            raise KeyError(f"Could not find the key '{one_missing_key}'")

        # if there is not any found key, return empty results
        if not found_keys:
            return [None for _ in query_keys]
        if subset_keys is not None and len(subset_keys) == 0:
            return [None if key not in found_keys else tuple() for key in query_keys]

        # make search space and faiss index
        if subset_keys is not None:
            temp_key_to_data = {k: self.key_to_data[k] for k in subset_keys}
            selected_idx_to_key, selected_index_ = self._make_index(key_to_data=temp_key_to_data)
            # max. len(subset_keys) ids may be retrieved
            n_results = min(len(subset_keys), n_results)
        else:
            selected_index_ = self.index_
            selected_idx_to_key = self.idx_to_key
            # just to be safe for very small collections
            n_results = min(self.get_count(), n_results)

        # search and find similar indices and keys
        query_embeddings = np.array(self.get_embeddings(found_keys, reshape=None))
        batch_similarities, batch_similar_indices = selected_index_.search(query_embeddings, n_results)
        batch_similar_keys = [
            [selected_idx_to_key[i] for i in row] for row in batch_similar_indices
        ]

        # make the formatted results
        found_keys = set(found_keys)  # for searching in O(1)
        projector = {
            "key": lambda i, j: batch_similar_keys[i][j],
            "similarity": lambda i, j: float(batch_similarities[i][j]),
            "embedding": lambda i, j: self.get_embedding(
                key=batch_similar_keys[i][j], reshape=None).tolist()
        }
        # pre-determine the output formatter with lambda fn (instead of using if-else in a loop)
        if return_field_names:
            results_fn = lambda i, j: {field: projector[field](i, j) for field in projection}
        else:
            results_fn = lambda i, j: tuple(projector[field](i, j) for field in projection)
        # if key doesn't exist --> None
        # if key exists --> for each query return a list of max.(*) n_results similars
        # if return_fields for each query key --> [{"key": "145", "similarity": 0.2, ...}, ...]
        # if not return_fields for each query key --> [("145", 0.2), ...]
        # *: if threshold is set and not satisifed over n_results similars, returned list is less than n_results
        i, results = 0, []
        for key in query_keys:
            if key not in found_keys:
                results.append(None)
                continue
            results.append([
                results_fn(i, j)
                for j in range(n_results)
                if threshold is None or batch_similarities[i][j] >= threshold
            ])
            i += 1
        return results

    def deserialize(self, data):
        """
        Deserializes a dictionary containing a FAISS index and metadata.

        Args:
            data (dict): A dictionary containing serialized FAISS index and metadata.

        Returns:
            self: The FaissIndex instance with the deserialized data.
        """
        self.index_ = faiss.deserialize_index(data["index_"])
        self.index_type = data["index_type"]
        self.index_metric = data["index_metric"]
        self.key_to_data = data["key_to_data"]
        self.idx_to_key = data["idx_to_key"]
        return self

    def serialize(self):
        """
        Serializes the Faiss index and metadata into a dictionary.

        Returns:
            dict: A dictionary containing the serialized Faiss index and metadata.
                  - "index_": Serialized FAISS index.
                  - "index_type": FAISS index type.
                  - "index_metric": Distance metric used in FAISS.
                  - "key_to_data": Dictionary mapping keys to their embeddings.
                  - "idx_to_key": Mapping of index positions to keys.
        """
        data = dict()
        data["index_"] = faiss.serialize_index(self.index_)
        data["index_type"] = self.index_type
        data["index_metric"] = self.index_metric
        data["key_to_data"] = self.key_to_data
        data["idx_to_key"] = self.idx_to_key
        return data

    def save(self, file_path):
        """
        Saves the serialized Faiss index and metadata to a file using pickle.

        Args:
            file_path (str): Path to the file where the index should be saved.

        Returns:
            self: The FaissIndex instance.
        """
        data = self.serialize()
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        self.logger.info(self._make_log({
            "msg": "FaissIndex successfully saved",
            "num_items": self.get_count(),
            "save_path": file_path}
        ))
        return self

    def back_up(self, folder_path):
        """
        Creates a backup of the FAISS database file in the specified folder.

        Args:
            folder_path (str): Path to the backup folder.

        Returns:
            self: The FaissIndex instance.
        """
        validate_folder(folder_path, not_empty_ok=True)
        shutil.copy2(self.db_path, folder_path)

        self.logger.info(self._make_log({
            "msg": "FaissIndex successfully backed up",
            "num_items": self.get_count(),
            "backup_folder": folder_path}
        ))
        return self

    def back_up_to_gcs(self, creds, bucket, folder_path):
        """
        Backs up the FAISS database file to Google Cloud Storage (GCS).

        Args:
            creds (str or dict): GCS credentials.
            bucket (str): Name of the GCS bucket.
            folder_path (str): Path within the GCS bucket where the backup should be stored.

        Returns:
            self: The FaissIndex instance.
        """
        validate_gcs_folder(creds, bucket, folder_path, not_empty_ok=True)
        upload_to_gcs(creds, self.db_path, bucket, folder_path)

        self.logger.info(self._make_log({
            "msg": "FaissIndex successfully backed up",
            "num_items": self.get_count(),
            "backup_folder": f'gs:/{bucket}/{folder_path}'}
        ))
        return self

    def rollback(self):
        """
        Rolls back any uncommitted changes to the FAISS index.
        This function is intended to be used if `self.add()` fails. It resets the index
        to the last committed state and clears any temporary data.

        Returns:
            self: The FaissIndex instance.
        """
        # add() should not require the below line, but still make index with saved data incase of a fail during commit
        self.idx_to_key, self.index_ = self._make_index()
        self._clear_stack()
        return self

    def commit(self, allow_empty_commit=False):
        """
        Commits pending changes made by `self.add()`, making them persistent.
        This function must be called to make changes effective. If no changes exist and
        `allow_empty_commit` is `False`, an error is raised.

        Args:
            allow_empty_commit (bool, optional): If `True`, allows committing even if
                there are no changes. Defaults to `False`.

        Returns:
            self: The FaissIndex instance.

        Raises:
            ValueError: If there are no pending changes and `allow_empty_commit` is `False`.
        """
        if self._is_stacked:
            self.key_to_data = copy.deepcopy(self._stacked_key_to_data)
            self.idx_to_key = copy.deepcopy(self._stacked_idx_to_key)
            self.index_ = faiss.clone_index(self._stacked_index_)
            self._clear_stack()
            self.save(self.db_path)  # write to disk for persistance
        else:
            if not allow_empty_commit:
                raise ValueError("There is no data change saved for a commit")
        return self

    def drop(self, not_exists_ok=False):
        """
        Deletes the FAISS index file from disk and resets in-memory data.
        If the FAISS index file exists, it will be removed. If it does not exist and
        `not_exists_ok` is `False`, an error is raised.

        Args:
            not_exists_ok (bool, optional): If `True`, does not raise an error if the index
                file does not exist. Defaults to `False`.

        Returns:
            self: The FaissIndex instance.

        Raises:
            ResourceNotFoundError: If the FAISS index file does not exist and `not_exists_ok` is `False`.
        """
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            self._reset_data()
        else:
            if not not_exists_ok:
                raise ResourceNotFoundError(f"FaissIndex at '{self.db_url}' does not exist")
        return self

    def _reset_data(self):
        """
        Resets the FAISS index and associated data.
        This method clears all stored data, resets the index type and metric,
        and removes the existing FAISS index. It also clears any temporary stacked data.

        Returns:
            None
        """
        self.key_to_data = dict()
        self.idx_to_key = dict()
        self.index_type = self.index_type
        self.index_metric = self.index_metric
        self.index_ = None
        # default stack variables
        self._clear_stack()

    def _make_index(self, key_to_data=None):
        """
        Creates a FAISS index using the provided key-to-data mapping.
        If `key_to_data` is `None`, it uses the full data (`self.key_to_data`).
        This method is used for both creating the main index and generating temporary indices
        for filtered searches.

        Args:
            key_to_data (dict, optional): A mapping of keys to embeddings. If `None`,
                uses `self.key_to_data`. Defaults to `None`.

        Returns:
            tuple: A dictionary mapping index positions to keys (`idx_to_key`),
                   and the FAISS index object (`index_`).
        """
        selected_key_to_data = key_to_data if key_to_data else self.key_to_data
        idx_to_key = {idx: key for idx, key in enumerate(list(selected_key_to_data.keys()))}
        embeddings = np.array(list(selected_key_to_data.values()))
        index_ = faiss.index_factory(embeddings.shape[1], self.index_type, getattr(faiss, self.index_metric))
        index_.add(embeddings)
        return idx_to_key, index_

    def _stack_data_change(self, key_to_data_for_stack):
        """
        Temporarily stores and updates a stacked version of the FAISS index before committing.
        This method creates a deep copy of the current data, updates it with the provided
        new data, and builds a temporary FAISS index with the changes. The stacked index
        is kept until a commit is made.

        Args:
            key_to_data_for_stack (dict): A mapping of new keys to embeddings
                that will be added to the stacked index.

        Returns:
            self: The FaissIndex instance.
        """
        self._stacked_key_to_data = copy.deepcopy(self.key_to_data)
        self._stacked_key_to_data.update(key_to_data_for_stack)
        self._stacked_idx_to_key, self._stacked_index_ = self._make_index(self._stacked_key_to_data)
        self._is_stacked = True
        return self

    def _clear_stack(self):
        """
        Clears the stacked (temporary) FAISS index and restores the original state.
        This method resets all stacked data structures and marks `_is_stacked` as `False`,
        stating no pending changes.

        Returns:
            self: The FaissIndex instance.
        """
        self._stacked_key_to_data = dict()
        self._stacked_idx_to_key = dict()
        self._stacked_index_ = None
        self._is_stacked = False
        return self

    def _make_log(self, extra={}):
        msg = {
            "extra": {"db_url": self.db_url, **extra}
        }
        return msg

    def clone(self, db_url, with_stack=True):
        """
        Creates a deep copy of the current FAISS index instance.
        This method clones all stored data, FAISS index properties, and optionally the stacked
        (uncommitted) data. The cloned instance is initialized with the given `db_url`.

        Args:
            db_url (str): The database URL for the cloned instance.
            with_stack (bool, optional): If `True`, the stacked data and index are also cloned.
                Defaults to `True`.

        Returns:
            FaissIndex: A new instance of FaissIndex with the same data and index properties.
        """
        cloned_instance = self.__class__(
            db_url=db_url
        )
        cloned_instance.key_to_data = copy.deepcopy(self.key_to_data)
        cloned_instance.idx_to_key = copy.deepcopy(self.idx_to_key)
        if self.index_ is not None:
            cloned_instance.index_ = faiss.clone_index(self.index_)
        if with_stack:
            if self._is_stacked:
                cloned_instance._stacked_key_to_data = copy.deepcopy(self._stacked_key_to_data)
                cloned_instance._stacked_idx_to_key = copy.deepcopy(self._stacked_idx_to_key)
                if self._stacked_index_ is not None:
                    cloned_instance._stacked_index_ = faiss.clone_index(self._stacked_index_)
                else:
                    cloned_instance._stacked_index_ = None
                cloned_instance._is_stacked = True
            else:
                cloned_instance._is_stacked = False
        return cloned_instance

    @staticmethod
    def equals(index1, index2, with_stack=True):
        """
        Compares two FaissIndex instances for equality.

        This method checks the following properties:
        1. Index type
        2. Index metric
        3. Key-to-index mapping (`idx_to_key`)
        4. Key-to-data dictionary (`key_to_data`)
        5. FAISS index properties (`index_`)
        6. If `with_stack=True`, it also compares stacked data and index.

        Args:
            index1 (FaissIndex): First FAISS index instance.
            index2 (FaissIndex): Second FAISS index instance.
            with_stack (bool, optional): If `True`, also compares stacked data.
                Defaults to `True`.

        Raises:
            AssertionError: If any properties differ between the two indices.
        """
        if index1.index_type != index2.index_type:
            raise AssertionError(f"Different index types: {index1.index_type} vs {index2.index_type}")
        if index1.index_metric != index2.index_metric:
            raise AssertionError(f"Different metrics: {index1.index_metric} vs {index2.index_metric}")
        if index1.idx_to_key != index2.idx_to_key:
            raise AssertionError(f"Different key mapping: {index1.idx_to_key} vs {index2.idx_to_key}")
        FaissIndex.compare_key_to_data(index1.key_to_data, index2.key_to_data)
        FaissIndex.index_property_equals(index1.index_, index2.index_)
        if with_stack:
            if index1._is_stacked is not index2._is_stacked:
                raise AssertionError("Different stack state")
            if index1._stacked_idx_to_key != index2._stacked_idx_to_key:
                raise AssertionError("Different stack key mapping")
            FaissIndex.compare_key_to_data(index1._stacked_key_to_data, index2._stacked_key_to_data)
            FaissIndex.index_property_equals(index1._stacked_index_, index2._stacked_index_)

    @staticmethod
    def index_property_equals(faiss_index1, faiss_index2):
        """
        Compares two FAISS index objects for equality.
        This method checks:
        1. Number of stored vectors (`ntotal`).
        2. Dimensionality of vectors (`d`).
        3. Index type.
        4. Index metric.
        5. Stored vectors (using `np.allclose` to check for floating-point similarity).

        Args:
            faiss_index1 (faiss.Index): First FAISS index.
            faiss_index2 (faiss.Index): Second FAISS index.

        Raises:
            AssertionError: If the indices have different properties or vector contents.
        """
        if faiss_index1 is None and faiss_index2 is None:
            return
        if faiss_index1.ntotal != faiss_index2.ntotal:
            raise AssertionError(f"Different number of vectors: {faiss_index1.ntotal} vs {faiss_index2.ntotal}")
        if faiss_index1.d != faiss_index2.d:
            raise AssertionError(f"Different dimensionality: {faiss_index1.d} vs {faiss_index2.d}")
        if not isinstance(faiss_index1, faiss.IndexFlat) or not isinstance(faiss_index2, faiss.IndexFlat):
            raise AssertionError(f"Different index types: {type(faiss_index1)} vs {type(faiss_index2)}")
        if faiss_index1.metric_type != faiss_index2.metric_type:
            raise AssertionError(f"Different metrics: {faiss_index1.metric_type} vs {faiss_index2.metric_type}")
        vectors1 = faiss_index1.reconstruct_n(0, faiss_index1.ntotal)
        vectors2 = faiss_index2.reconstruct_n(0, faiss_index2.ntotal)
        if not np.allclose(vectors1, vectors2):
            raise AssertionError("Vectors are different between the indices.")

    @staticmethod
    def compare_key_to_data(key_embedding_dict1, key_embedding_dict2, atol=1e-8, rtol=1e-5):
        """
        Compares two dictionaries of key-to-embedding mappings.

        This method ensures that:
        - Both dictionaries contain the same keys.
        - Embeddings for corresponding keys are numerically similar within `atol` (absolute tolerance)
          and `rtol` (relative tolerance).

        Args:
            key_embedding_dict1 (dict): First key-to-embedding mapping.
            key_embedding_dict2 (dict): Second key-to-embedding mapping.
            atol (float, optional): Absolute tolerance for numerical comparison. Defaults to `1e-8`.
            rtol (float, optional): Relative tolerance for numerical comparison. Defaults to `1e-5`.

        Raises:
            AssertionError: If the keys differ or if any embeddings do not match within tolerance.
        """
        keys1 = set(key_embedding_dict1.keys())
        keys2 = set(key_embedding_dict2.keys())
        if keys1 != keys2:
            raise AssertionError(f"{len(keys1.symmetric_difference(keys2))} different keys found")
        for key in keys1:
            embedding1 = np.array(key_embedding_dict1[key])
            embedding2 = np.array(key_embedding_dict2[key])
            if not np.allclose(embedding1, embedding2, atol=atol, rtol=rtol):
                raise AssertionError(f"Different embeddings for key '{key}'")

    @staticmethod
    def _format_keys(keys):
        """
        Ensures that the provided keys are in a list-compatible format.
        If `keys` is not a list or iterable, it converts it into a list.

        Args:
            keys (Any): A single key or a list of keys.

        Returns:
            list: A list of keys.
        """
        if not is_list_compatible(keys):
            keys = [keys]
        return keys
