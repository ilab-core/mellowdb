import json
import logging

import grpc

import mellow_db.protocols.service_pb2 as pb2
import mellow_db.protocols.service_pb2_grpc as pb2_grpc
from mellow_db.converter import from_flex_item, to_flex_item
from mellow_db.decorators import response_handler
from mellow_db.exceptions import ActiveConnectionFound, ConnectionFailedError
from mellow_db.log import setup_client_logging
from mellow_db.utils import get_client_credentials

setup_client_logging()


class MellowClient:
    """
    A client for interacting with the Mellow service via gRPC.
    This class allows you to connect to the Mellow service, create collections,
    and use collections for further operations. It supports secure gRPC communication
    with configurable timeouts and message sizes.

    Attributes:
        logger (logging.Logger): Logger for the MellowClient class.
        channel (grpc.Channel): gRPC channel used for communication with the Mellow service.
        stub (pb2_grpc.MellowServiceStub): gRPC service stub for making requests to the Mellow service.
        collection_name (str): Name of the currently active collection (if any).
        client_id (str): Unique client identifier assigned after successful connection.
    """

    def __init__(self, host, port, service_account_info,
                 connect_timeout=90,
                 channel_options=[
                     # Maximum send message size (1 GB) for gRPC communication
                     ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                     # Maximum receive message size (1 GB) for gRPC communication
                     ("grpc.max_receive_message_length", 1024 * 1024 * 1024)]):
        """
        Initialize the MellowClient with the necessary configurations and
        establish a connection to the Mellow service.

        Args:
            host (str): The host address of the Mellow service.
            port (int): The port number to connect to on the Mellow service.
            service_account_info (dict): Service account information for authenticating the client.
            connect_timeout (int): Timeout duration for the connection.
                Defaults to 90 seconds.
            channel_options (list): Optional additional gRPC channel options.
                Defaults to maximum send and receive message sizes of 1 GB.

        Raises:
            ConnectionFailedError: If the connection to the Mellow service fails.
        """
        self.logger = logging.getLogger("MellowClient")
        credentials = get_client_credentials(service_account_info)
        default_options = [
            ('grpc.keepalive_time_ms', 60000),  # Ping the server every 60 seconds
            ('grpc.keepalive_timeout_ms', 30000),  # Wait 30 seconds for a ping response
            ('grpc.keepalive_permit_without_calls', True),  # Allow keepalives when no calls are active
            ('grpc.http2.max_pings_without_data', 0),  # Allow unlimited pings without data
            ('grpc.http2.min_time_between_pings_ms', 60000),  # Minimum 60 seconds between pings
            ('grpc.http2.max_ping_strikes', 5),  # Termination due to 5 missed pings,
        ]
        self.channel = grpc.secure_channel(f"{host}:{port}",
                                           credentials,
                                           options=default_options + channel_options)
        self.stub = pb2_grpc.MellowServiceStub(self.channel)
        self.collection_name = None

        # connect
        response = self.stub.connect(pb2.ConnectRequest(), timeout=connect_timeout)
        self.logger.info(response.message)
        if not response.client_id:
            raise ConnectionFailedError("Could not assign client_id. Connection failed")

        self.client_id = response.client_id
        self.logger.debug(f"Assigned client_id: '{self.client_id}'")

    @response_handler(collection_name_check=False)
    def create_collection(self, collection_name, index_config={}, collection_schema=[]):
        """
        Create a new collection in the Mellow service with the specified name,
        index configuration, and matadata schema.

        Args:
            collection_name (str): The name of the collection to create.
            index_config (dict): Index configuration parameters. Defaults to an empty dictionary.
            collection_schema (list): Metadata schema for the collection. Defaults to an empty list.

        Returns:
            str: The response message from the Mellow service.
        """
        create_request_params = {
            "collection_name": collection_name,
            "index_config": self._make_index_config_request(index_config),
            "collection_schema": self._make_schema_request(collection_schema),
            "client_id": self.client_id
        }
        request = pb2.CreateCollectionRequest(**create_request_params)
        response = self.stub.create_collection(request)
        self.logger.info(response.message)
        return response.message

    @response_handler(collection_name_check=False)
    def use_collection(self, collection_name):
        """
        Set the currently active collection for the client. If a collection is
        already in use, raises an exception to prevent multiple active connections.

        Args:
            collection_name (str): The name of the collection to use.

        Returns:
            str: The response message from the Mellow service.

        Raises:
            ActiveConnectionFound: If the client is already connected to a collection.
        """
        if self.collection_name:
            raise ActiveConnectionFound((f"Client already has an active connection to collection '{self.collection_name}'. "
                                         "Close the active connection first with disconnect()"))

        use_request_params = {"collection_name": collection_name, "client_id": self.client_id}
        request = pb2.UseCollectionRequest(**use_request_params)
        response = self.stub.use_collection(request)
        self.logger.info(response.message)

        self.collection_name = collection_name
        self.logger.debug(f"Using collection: '{self.collection_name}'")
        return response.message

    @response_handler()
    def get_collection_item_count(self):
        """
        Retrieve the item count of the currently active collection.

        Returns:
            int: The total number of items in the collection.
        """
        request = pb2.GetCollectionItemCountRequest(client_id=self.client_id)
        response = self.stub.get_collection_item_count(request)
        return response.item_count

    @response_handler()
    def get_collection_info(self):
        """
        Retrieve detailed information about the currently active collection.

        Returns:
            dict: A dictionary containing information about the collection, including:
                - name (str): The name of the collection.
                - item_count (int): The number of items in the collection.
                - size_in_bytes (int): The total size of the collection in bytes.
                - faiss_index_type (str): The FAISS index type used.
                - faiss_index_metric (str): The FAISS index metric used.
                - embedding_dim (int): The dimensionality of the embeddings.
                - primary_keys (list): List of primary key field names.
                - meta_columns (list): Metadata columns in the collection,
                    with their 'name', 'type', 'is_nullable', 'is_index', 'default' information.
        """
        request = pb2.GetCollectionInfoRequest(client_id=self.client_id)
        response = self.stub.get_collection_info(request)
        info = {
            "name": self.collection_name,
            "item_count": response.item_count,
            "size_in_bytes": response.size_in_bytes,
            "faiss_index_type": response.faiss_index_type,
            "faiss_index_metric": response.faiss_index_metric,
            "embedding_dim": response.embedding_dim,
            "primary_keys": response.primary_keys,
            "meta_columns": [{info.key: from_flex_item(info.value) for info in col_info.items}
                             for col_info in response.meta_columns
                             ]
        }
        return info

    @response_handler()
    def add(self, key_to_data=None, key_to_metadata=None, upsert=False):
        """
        Add or update items in the currently active collection.
        This method allows you to insert or update items in the collection, either by
        providing both data and metadata or just one of them.

        Args:
            key_to_data (dict, optional): A dictionary mapping keys to data values (embeddings).
            key_to_metadata (dict, optional): A dictionary mapping keys to metadata values.
            upsert (bool, optional): Whether to update existing items (default is False).

        Returns:
            int: The updated item count in the collection.

        Raises:
            ValueError: If neither `key_to_data` nor `key_to_metadata` is provided.
            ValueError: If the lengths or keys of `key_to_data` and `key_to_metadata` don't match.
        """
        if not (key_to_data or key_to_metadata):
            raise ValueError("One or both of the fields 'key_to_data' and 'key_to_metadata' must be given")
        if key_to_data and key_to_metadata:
            length_key_to_data = len(key_to_data)
            length_key_to_metadata = len(key_to_metadata)
            if length_key_to_data != length_key_to_metadata:
                raise ValueError("Field 'key_to_data' and 'key_to_metadata' lengths must match! "
                                 f"Found {length_key_to_data} and {length_key_to_metadata}")
            diff_keys = set(key_to_data.keys()).difference(set(key_to_metadata.keys()))
            if diff_keys:
                raise ValueError("Field 'key_to_data' and 'key_to_metadata' keys must match! "
                                 f"Found different {len(diff_keys)} keys")

        add_request_params = {"client_id": self.client_id, "upsert": upsert}
        if key_to_data is not None:
            add_request_params["embedding"] = [
                pb2.KeyedEmbedding(key=key, embedding=values) for key, values in key_to_data.items()
            ]
        if key_to_metadata is not None:
            add_request_params["metadata"] = MellowClient._make_metadata_or_filter_request(key_to_metadata)

        request = pb2.AddRequest(**add_request_params)
        response = self.stub.add(request)
        return response.item_count

    @response_handler()
    def get(self, where=None, projection=None, limit=None):
        """
        Retrieve items from the currently active collection based on the given filters.

        Args:
            where (dict, optional): A dictionary of filters for selecting specific items.
            projection (list, optional): A list of fields to return.
                Default returns only 'key' field.
            limit (int, optional): The maximum number of results to return. Default is no limit.

        Returns:
            list: A list of tuples representing the matching items in the collection.

        Raises:
            ValueError: If `projection` is an empty list or `limit` is set to 0.
        """
        if projection == []:
            raise ValueError("Projection can not be empty. Pass None or ['key'] for only returning primary keys")
        if limit == 0:
            raise ValueError("'limit' must be greater than 0 or None for no limit")
        if not where and not limit:
            self.logger.warning("No 'where' or 'limit' provided. This will return all rows. It may take a while")

        get_request_params = {"client_id": self.client_id}
        if where is not None:
            get_request_params["where"] = MellowClient._make_metadata_or_filter_request(where)
        if projection is not None:
            get_request_params["projection"] = projection
        if limit is not None:
            get_request_params["limit"] = limit

        request = pb2.GetRequest(**get_request_params)
        response = self.stub.get(request)
        response = [tuple(from_flex_item(col) for col in row.items) for row in response.results]
        return response

    @response_handler()
    def search(self, query_keys, where=None, projection=None, n_results=None, threshold=None, not_exists_ok=False):
        """
        Search for items in the collection based on the provided query keys and optional filters on the search space.
        This method performs a search in the currently active collection using the provided `query_keys`
        and applies additional filters (e.g., `where`, `projection`, `n_results`, `threshold`)
        to the search space to limit the results.

        Args:
            query_keys (list): A list of keys to query in the collection.
            where (dict, optional): A dictionary of filters to apply to the search space.
                Uses MongoDB-like syntax, e.g., {"field_name": {"$eq": value}}.
            projection (list, optional): A list of fields to return.
                Defaults to returning only the 'key' and 'similarity' fields.
            n_results (int, optional): The maximum number of results to return. Must be greater than 0.
            threshold (float, optional): A threshold for filtering results based on similarity.
            not_exists_ok (bool, optional): Whether to allow searches for non-existent items.
                Defaults to False.

        Returns:
            list: A list of search results, where each search result is a tuple of tuples.
                Each inner tuple contains the fields specified in `projection`,
                in the order they were requested.

        Raises:
            ValueError: If `query_keys` is empty or not a list,
                if `n_results` is less than 1,
                or if `projection` is an empty list.
        """
        if projection == []:
            raise ValueError("Projection can not be empty. Pass None or ['key'] for only returning primary keys")
        if n_results < 1:
            raise ValueError("'n_results' must be grreater than 0")
        if not len(query_keys):
            raise ValueError("'query_keys' can not be empty")
        if not isinstance(query_keys, list):
            raise ValueError(f"'query_keys' must be a list instance. Found '{type(query_keys)}'")

        search_request_params = {"client_id": self.client_id, "query_keys": query_keys, "not_exists_ok": not_exists_ok}
        if where is not None:
            search_request_params["where"] = MellowClient._make_metadata_or_filter_request(where)
        if projection is not None:
            search_request_params["projection"] = projection
        if n_results is not None:
            search_request_params["n_results"] = n_results
        if threshold is not None:
            search_request_params["threshold"] = threshold

        request = pb2.SearchRequest(**search_request_params)
        response = self.stub.search(request)
        response = [
            None if row.is_null else tuple(tuple(from_flex_item(col) for col in similarity.items)
                                           for similarity in row.items)
            for row in response.results
        ]
        return response

    @response_handler(collection_name_check=False)
    def delete_collection(self, collection_name):
        """
        Delete the specified collection from the server.

        Args:
            collection_name (str): The name of the collection to delete.

        Returns:
            str: A message indicating the result of the operation.
        """
        request = pb2.DeleteCollectionRequest(collection_name=collection_name,
                                              client_id=self.client_id)
        response = self.stub.delete_collection(request)
        self.logger.info(response.message)
        self.collection_name = None
        return response.message

    @response_handler(collection_name_check=False)
    def disconnect(self):
        """
        Disconnect the client from the server and close the current connection.
        This method sends a request to the Mellow service to disconnect the client,
        invalidates the client ID, and closes the communication channel.

        Returns:
            str: A message indicating the result of the disconnection operation.
        """
        request = pb2.DisconnectRequest(client_id=self.client_id)
        response = self.stub.disconnect(request)
        self.logger.info(response.message)

        self.client_id = None
        self.collection_name = None
        self.channel.close()
        self.logger.debug(("Disconnected from the server.\n"
                          f"client_id: '{self.client_id}' collection: '{self.collection_name}'"))
        return response.message

    @response_handler()
    def eval(self, subset_size, test_size, k=30, tolerance=1e-5):
        """
        Evaluate the performance of the current collection using a subset of items.
        This method evaluates the collection's performance by testing it on a subset of items
        and calculating performance metrics such as precision, recall, and search time.

        Args:
            subset_size (int): The number of items to include in the evaluation subset.
            test_size (int): The number of test items to evaluate.
            k (int, optional): The number of nearest neighbors to consider (default: 30).
            tolerance (float, optional): The tolerance level for evaluation (default: 1e-5).

        Returns:
            tuple: A tuple containing the following performance metrics:
                - avg_mellow_time (float): Average search time using MellowDB.
                - avg_cosine_time (float): Average search time using scikit-learn's cosine similarity.
                - avg_precision (float): Average precision comparing MellowDB and scikit-learn results.
                - avg_recall (float): Average recall comparing MellowDB and scikit-learn results.
                - sum_diffs (float): Sum of differences between MellowDB and scikit-learn results.
                - avg_diffs (float): Average difference between MellowDB and scikit-learn results.
        """
        request = pb2.EvalRequest(
            client_id=self.client_id,
            subset_size=subset_size,
            test_size=test_size,
            k=k,
            tolerance=tolerance
        )
        response = self.stub.eval(request)
        return (
            response.avg_mellow_time,
            response.avg_cosine_time,
            response.avg_precision,
            response.avg_recall,
            response.sum_diffs,
            response.avg_diffs
        )

    @response_handler()
    def back_up(self, backup_dir):
        """
        Create a backup of the current collection.
        This method saves a backup of the currently active collection to the specified directory.

        Args:
            backup_dir (str): The directory where the backup should be stored.

        Returns:
            tuple: A tuple containing:
                - message (str): A response message indicating the status of the backup.
                - backup_full_path (str): The full path to the created backup file.
        """
        request = pb2.BackUpRequest(client_id=self.client_id, backup_dir=backup_dir)
        response = self.stub.back_up(request)
        return response.message, response.backup_full_path

    @response_handler()
    def back_up_to_gcs(self, creds, bucket, backup_dir):
        """
        Create a backup of the current collection and upload it to Google Cloud Storage (GCS).
        This method saves a backup of the currently active collection and uploads it to the specified GCS bucket.

        Args:
            creds (dict or str): Google Cloud credentials as a dictionary or a JSON string.
            bucket (str): The name of the GCS bucket where the backup will be stored.
            backup_dir (str): The directory path within the bucket where the backup should be saved.

        Returns:
            tuple: A tuple containing:
                - message (str): A response message indicating the status of the backup.
                - backup_full_path (str): The full path to the backup file in GCS.
        """
        # Convert to a JSON string, if creds is a dictionary
        if isinstance(creds, dict):
            creds = json.dumps(creds)

        request = pb2.BackUpToGcsRequest(client_id=self.client_id,
                                         creds=creds,
                                         bucket=bucket,
                                         backup_dir=backup_dir)
        response = self.stub.back_up_to_gcs(request)
        return response.message, response.backup_full_path

    @response_handler()
    def load_from_path(self, path, collection_name):
        """
        Load a collection from a local backup file.
        This method restores a collection from a backup file stored at the specified local path.

        Args:
            path (str): The file path to the backup.
            collection_name (str): The name of the collection to be restored.

        Returns:
            str: A response message indicating the status of the process.
        """
        request = pb2.LoadFromPathRequest(client_id=self.client_id,
                                          path=path,
                                          collection_name=collection_name)
        response = self.stub.load_from_path(request)
        return response.message

    @response_handler()
    def load_from_gcs(self, creds, bucket, path, collection_name):
        """
        Load a collection from a backup stored in Google Cloud Storage (GCS).
        This method restores a collection from a backup file stored in a GCS bucket.

        Args:
            creds (dict or str): Google Cloud credentials as a dictionary or a JSON string.
            bucket (str): The name of the GCS bucket where the backup is stored.
            path (str): The file path to the backup within the bucket.
            collection_name (str): The name of the collection to be restored.

        Returns:
            str: A response message indicating the status of the process.
        """
        # Convert to a JSON string, if creds is a dictionary
        if isinstance(creds, dict):
            creds = json.dumps(creds)

        request = pb2.LoadFromGcsRequest(client_id=self.client_id,
                                         creds=creds,
                                         bucket=bucket,
                                         path=path,
                                         collection_name=collection_name)
        response = self.stub.load_from_gcs(request)
        return response.message

    @staticmethod
    def _make_index_config_request(config):
        return pb2.KeyedFlexItemList(
            items=[pb2.KeyedFlexItem(key=key, value=to_flex_item(value))
                   for key, value in config.items()])

    @staticmethod
    def _make_schema_request(schema):
        return [
            pb2.CollectionSchema(
                field_name=schema_item["field_name"],
                field_type=schema_item["field_type"],
                is_nullable=schema_item["is_nullable"])
            for schema_item in schema
        ]

    @staticmethod
    def _make_metadata_or_filter_request(flex_map_list):
        return [
            pb2.KeyedKeyedFlexList(
                key=key,
                items=[pb2.KeyedFlexItem(key=inner_key, value=to_flex_item(value))
                       for inner_key, value in flex_map.items()])
            for key, flex_map in flex_map_list.items()
        ]
