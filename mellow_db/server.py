import datetime as dt
import json
import logging
import signal
import sys
import uuid
from concurrent import futures

import grpc
import numpy as np

import mellow_db.protocols.service_pb2 as pb2
import mellow_db.protocols.service_pb2_grpc as pb2_grpc
from mellow_db.collection import Collection
from mellow_db.connection_manager import ConnectionManager
from mellow_db.converter import from_flex_item, to_flex_item
from mellow_db.decorators import request_handler
from mellow_db.exceptions import (CollectionHasActiveClientsError,
                                  CollectionNotExistsError,
                                  ConnectionTimedOutError, InvalidRequest)
from mellow_db.log import setup_server_logging
from mellow_db.utils import get_server_credentials, load_yaml


class MellowServer(pb2_grpc.MellowServiceServicer):
    """
    gRPC server implementation for MellowDB, handling client connections and collection management.
    This class provides functionality for:
    - Connecting and disconnecting clients.
    - Handling DB operations.
    - Managing client assignments, disconnection and thread-safe collection operations.

    Attributes:
        logger (logging.Logger): Logger instance for the server.
        idle_timeout (int): Time (in seconds) before an inactive client is disconnected.
        cm (ConnectionManager): Manages client connections and active collections.
    """

    def __init__(self):
        """
        Initializes the MellowServer instance.

        Sets up logging, defines the inactivity timeout for clients, and initializes the connection manager.
        """
        self.logger = logging.getLogger("server")
        self.idle_timeout = 2 * 60 * 60  # inactivity timeout is 2 hours
        self.cm = ConnectionManager(idle_timeout=self.idle_timeout)

    @request_handler(pb2.ConnectResponse, client_id_check=False, collection_name_check=False, reset_idle_timer=False)
    def connect(self, request, context):
        """
        Handles a client connection request.
        Assigns a new unique client ID and registers the client in the connection manager.

        Args:
            request (pb2.ConnectRequest): The gRPC request.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.ConnectResponse: Response containing the assigned client ID.
        """
        client_id = str(uuid.uuid4())
        self.cm.add_client(client_id)

        self.logger.debug(self._make_log(client_id, None))

        return pb2.ConnectResponse(message="Client assigned successfully", client_id=client_id)

    @request_handler(pb2.DisconnectResponse, collection_name_check=False, reset_idle_timer=False)
    def disconnect(self, request, context):
        """
        Handles a client disconnection request.
        Removes the client gracefully via the connection manager.

        Args:
            request (pb2.DisconnectRequest): The gRPC request containing the client ID.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.DisconnectResponse: Response confirming the disconnection.
        """
        collection_name = self.cm.remove_client(request.client_id)

        self.logger.debug(self._make_log(request.client_id, collection_name))

        return pb2.DisconnectResponse(message="Disconnected successfully")

    @request_handler(pb2.CreateCollectionResponse, collection_name_check=False)
    def create_collection(self, request, context):
        """
        Handles a request to create a new collection.
        Temporarily enables write mode, creates the collection with the specified schema and index configuration,
        and ensures write mode is disabled even if an error occurs.

        Args:
            request (pb2.CreateCollectionRequest): The gRPC request containing the client ID
                and collection details.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.CreateCollectionResponse: Response confirming the collection creation.
        """
        collection_name = request.collection_name
        self.cm.set_collection_write_mode(collection_name, in_write_mode=True)
        try:
            Collection.create(
                name=request.collection_name,
                schema=MellowServer._get_schema_request(request.collection_schema),
                index_config=MellowServer._get_index_config_request(request.index_config)
            )
        finally:
            # Always set back to read mode, even if an error occurs
            self.cm.set_collection_write_mode(collection_name, in_write_mode=False)

        self.logger.debug(self._make_log(request.client_id, request.collection_name))

        return pb2.CreateCollectionResponse(
            message=f"Created collection '{request.collection_name}' successfully")

    @request_handler(pb2.UseCollectionResponse, collection_name_check=False)
    def use_collection(self, request, context):
        """
        Assigns a collection to a client.
        Loads the collection into memory and assigns it to the specified client.

        Args:
            request (pb2.UseCollectionRequest): The gRPC request containing the client ID
                and the collection name.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.UseCollectionResponse: Response confirming the collection assignment.
        """
        self.cm.load_collection_to_memory(request.collection_name)
        self.cm.assign_collection(request.client_id, request.collection_name)

        self.logger.debug(self._make_log(request.client_id, request.collection_name))

        return pb2.UseCollectionResponse(message=f"Assigned collection '{request.collection_name}' successfully")

    @request_handler(pb2.DeleteCollectionResponse, collection_name_check=False)
    def delete_collection(self, request, context):
        """
        Handles a request to delete a collection.
        Ensures there are no active clients using the collection before deletion.
        Temporarily enables write mode for deletion and ensures write mode is disabled afterward.

        Args:
            request (pb2.DeleteCollectionRequest): The gRPC request containing the client ID
                and the collection name.
            context (grpc.ServicerContext): The gRPC context.

        Raises:
            CollectionHasActiveClientsError: If there are active clients connected to the collection.

        Returns:
            pb2.DeleteCollectionResponse: Response confirming the deletion.
        """
        collection_name = request.collection_name
        self._validate_collection(collection_name=collection_name)
        active_clients = self.cm.get_clients_of_collection(collection_name)
        if active_clients:
            raise CollectionHasActiveClientsError(
                f"There are {len(active_clients)} active clients connected to collection '{collection_name}'")

        collection_ = self.cm.load_collection_to_memory(collection_name)
        self.cm.set_collection_write_mode(collection_.name, in_write_mode=True)
        try:
            collection_.delete(not_exists_ok=False)
        finally:
            # Always set back to read mode, even if an error occurs
            self.cm.set_collection_write_mode(collection_.name, in_write_mode=False)

        self.logger.debug(self._make_log(request.client_id, request.collection_name))

        return pb2.DeleteCollectionResponse(message=f"Deleted collection '{collection_name}' successfully")

    @request_handler(pb2.GetCollectionItemCountResponse)
    def get_collection_item_count(self, request, context):
        """
        Retrieves the total number of items in the currently assigned collection for the client.

        Args:
            request (pb2.GetCollectionItemCountRequest): The gRPC request containing the client ID.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.GetCollectionItemCountResponse: Response containing the total item count.
        """
        collection_ = self.cm.get_collection(request.client_id)
        self.cm.wait_until_collection_available(collection_.name)
        item_count = collection_.get_count()

        self.logger.debug(self._make_log(request.client_id, collection_.name, extra={"count": item_count}))

        return pb2.GetCollectionItemCountResponse(item_count=item_count)

    @request_handler(pb2.GetCollectionInfoResponse)
    def get_collection_info(self, request, context):
        """
        Retrieves metadata and statistics about the currently assigned collection.

        Args:
            request (pb2.GetCollectionInfoRequest): The gRPC request containing the client ID.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.GetCollectionInfoResponse: Response containing collection's metadata.
        """
        collection_ = self.cm.get_collection(request.client_id)
        self.cm.wait_until_collection_available(collection_.name)
        info = collection_.get_info()

        self.logger.debug(self._make_log(request.client_id, collection_.name, extra={"info": info}))

        return pb2.GetCollectionInfoResponse(
            item_count=info["item_count"],
            size_in_bytes=info["size_in_bytes"],
            faiss_index_type=info["faiss_index_type"],
            faiss_index_metric=info["faiss_index_metric"],
            embedding_dim=info["embedding_dim"],
            primary_keys=info["primary_keys"],
            meta_columns=[pb2.KeyedFlexItemList(
                items=[pb2.KeyedFlexItem(key=key, value=to_flex_item(value)) for key, value in col_info.items()])
                for col_info in info["meta_columns"]]
        )

    @request_handler(pb2.AddResponse)
    def add(self, request, context):
        """
        Adds new items to the currently assigned collection.
        Rolled-back if an error occurs during addition.

        Args:
            request (pb2.AddRequest): The gRPC request containing the client ID
                and embeddings and metadata to be added.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            pb2.AddResponse: Response containing the updated item count in the collection.
        """
        params = {"upsert": request.upsert}
        if request.embedding:
            params["key_to_data"] = {
                item.key: np.array(item.embedding) for item in request.embedding
            }
        if request.metadata:
            params["key_to_metadata"] = MellowServer._get_metadata_or_filter_request(request.metadata)

        collection_ = self.cm.get_collection(request.client_id)
        self.cm.set_collection_write_mode(collection_.name, in_write_mode=True)
        try:
            init_count = collection_.get_count()  # for debugging
            collection_.add(**params)
            count = collection_.get_count()
        finally:
            # Always set back to read mode, even if an error occurs
            self.cm.set_collection_write_mode(collection_.name, in_write_mode=False)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"init_count": init_count, "updated_count": count}))

        return pb2.AddResponse(item_count=count)

    @request_handler(pb2.GetResponse)
    def get(self, request, context):
        """
        Retrieves items from the collection based on the provided filters.

        Args:
            request (pb2.GetRequest): The request containing the client ID and optional filters:
                - projection (list): List of fields to include in the result.
                - where (dict): MongoDB-like filtering conditions for metadata.
                - limit (int): Maximum number of items to retrieve.
            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.GetResponse: A response containing a list of retrieved items.
        """
        params = {}
        if request.projection:
            params["projection"] = request.projection
        if request.where:
            params["where"] = MellowServer._get_metadata_or_filter_request(request.where)
        if request.limit:
            params["limit"] = request.limit

        collection_ = self.cm.get_collection(request.client_id)
        self.cm.wait_until_collection_available(collection_.name)
        results = collection_.get(**params)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"result_length": len(results)}))

        return pb2.GetResponse(results=[
            pb2.FlexItemList(items=[to_flex_item(col_val) for col_val in row]) for row in results]
        )

    @request_handler(pb2.SearchResponse)
    def search(self, request, context):
        """
        Searches for similar items in the collection based on query keys.

        Args:
            request (pb2.SearchRequest): The search request containing the client ID and:
                - query_keys (list): List of keys to search for.
                - not_exists_ok (bool): Whether missing keys should be ignored.
                - where (dict, optional): MongoDB-like filtering conditions for metadata.
                - projection (list, optional): Fields to include in results.
                - n_results (int, optional): Number of top results to return.
                - threshold (float, optional): Minimum similarity score.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.SearchResponse: A response containing lists of similar items for each query.
        """
        params = {
            "query_keys": request.query_keys,
            "not_exists_ok": request.not_exists_ok
        }
        if request.where:
            params["where"] = MellowServer._get_metadata_or_filter_request(request.where)
        if request.projection:
            params["projection"] = request.projection
        if request.n_results:
            params["n_results"] = request.n_results
        if request.threshold:
            params["threshold"] = request.threshold

        collection_ = self.cm.get_collection(request.client_id)
        self.cm.wait_until_collection_available(collection_.name)
        results = collection_.search(**params)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"result_length": len(results)}))

        return pb2.SearchResponse(
            results=[pb2.FlexItemListList(
                is_null=(result is None),
                items=[pb2.FlexItemList(
                    items=[to_flex_item(similar) for similar in similars])
                    for similars in result]
                if result is not None else [])
                for result in results]
        )

    @request_handler(pb2.EvalResponse)
    def eval(self, request, context):
        """
        Evaluates the collection's search performance using a test set.

        Args:
            request (pb2.EvalRequest): The evaluation request containing the client ID and:
                - subset_size (int): The number of items to include in the evaluation subset.
                - test_size (int): The number of test items to evaluate.
                - k (int): The number of nearest neighbors to consider.
                - tolerance (float, optional): The tolerance level for evaluation.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.EvalResponse: A response containing evaluation metrics:
                - avg_mellow_time (float): Average search time using MellowDB.
                - avg_cosine_time (float): Average search time using scikit-learn's cosine similarity.
                - avg_precision (float): Average precision comparing MellowDB and scikit-learn results.
                - avg_recall (float): Average recall comparing MellowDB and scikit-learn results.
                - sum_diffs (float): Sum of differences between MellowDB and scikit-learn results.
                - avg_diffs (float): Average difference between MellowDB and scikit-learn results.
        """
        params = {
            "subset_size": request.subset_size,
            "test_size": request.test_size,
            "k": request.k,
        }
        if request.tolerance:
            params["tolerance"] = request.tolerance

        collection_ = self.cm.get_collection(request.client_id)
        self.cm.wait_until_collection_available(collection_.name)
        result = collection_.eval(**params)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"result": list(result)}))

        return pb2.EvalResponse(
            avg_mellow_time=result[0],
            avg_cosine_time=result[1],
            avg_precision=result[2],
            avg_recall=result[3],
            sum_diffs=result[4],
            avg_diffs=result[5]
        )

    @request_handler(pb2.BackUpResponse)
    def back_up(self, request, context):
        """
        Creates a local backup of the specified collection.

        Args:
            request (pb2.BackUpRequest): The backup request containing the client ID and:
                - backup_dir (str): The directory where the backup should be stored.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.BackUpResponse: A response containing:
                - message (str): A success message with the backup path.
                - backup_full_path (str): The full path of the backup file.
        """
        collection_ = self.cm.get_collection(request.client_id)
        backup_name = f"{collection_.name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
        self.cm.wait_until_collection_available(collection_.name)
        backup_full_path = collection_.back_up(backup_dir=request.backup_dir,
                                               backup_name=backup_name)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"backup_dir": request.backup_dir,
                                                "backup_name": backup_name}))

        return pb2.BackUpResponse(
            message=(f"Collection '{collection_.name}' backed up succesfully to "
                     f"{request.backup_dir}/{backup_name}"),
            backup_full_path=backup_full_path)

    @request_handler(pb2.BackUpToGcsResponse)
    def back_up_to_gcs(self, request, context):
        """
        Creates a backup of the specified collection and uploads it to Google Cloud Storage (GCS).

        Args:
            request (pb2.BackUpToGcsRequest): The backup request containing the client ID and:
                - creds (str): JSON-formatted credentials for GCS access.
                - bucket (str): The GCS bucket where the backup will be stored.
                - backup_dir (str): The directory path inside the bucket.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.BackUpToGcsResponse: A response containing:
                - message (str): A success message with the backup path in GCS.
                - backup_full_path (str): The full path of the backup file in GCS.
        """
        collection_ = self.cm.get_collection(request.client_id)
        backup_name = f"{collection_.name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
        creds = json.loads(request.creds)
        self.cm.wait_until_collection_available(collection_.name)
        backup_full_path = collection_.back_up_to_gcs(creds=creds,
                                                      bucket=request.bucket,
                                                      backup_dir=request.backup_dir,
                                                      backup_name=backup_name)

        self.logger.debug(self._make_log(request.client_id, collection_.name,
                                         extra={"bucket": request.bucket,
                                                "backup_dir": request.backup_dir,
                                                "backup_name": backup_name}))

        return pb2.BackUpToGcsResponse(
            message=(f"Collection '{collection_.name}' backed up succesfully to "
                     f"{request.backup_dir}/{request.bucket}/{backup_name}"),
            backup_full_path=backup_full_path)

    @request_handler(pb2.LoadFromPathResponse)
    def load_from_path(self, request, context):
        """
        Loads a collection from a local backup file.

        Args:
            request (pb2.LoadFromPathRequest): The load request containing the client ID and:
                - collection_name (str): The name of the collection to load.
                - path (str): The file path from which to load the collection.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.LoadFromPathResponse: A response containing:
                - message (str): A success message indicating the collection has been loaded.
        """
        collection_name = request.collection_name
        self.cm.set_collection_write_mode(collection_name, in_write_mode=True)
        try:
            Collection.load_from_path(collection_name, request.path)
        finally:
            # Always set back to read mode, even if an error occurs
            self.cm.set_collection_write_mode(collection_name, in_write_mode=False)

        self.logger.debug(self._make_log(request.client_id, request.collection_name))

        return pb2.LoadFromPathResponse(
            message=(f"Collection '{request.collection_name}' successfully loaded from"
                     f"{request.path}"))

    @request_handler(pb2.LoadFromGcsResponse)
    def load_from_gcs(self, request, context):
        """
        Loads a collection from a backup stored in Google Cloud Storage (GCS).

        Args:
            request (pb2.LoadFromGcsRequest): The load request containing the client ID and:
                - collection_name (str): The name of the collection to load.
                - creds (str): JSON-formatted credentials for GCS access.
                - bucket (str): The GCS bucket where the backup is stored.
                - path (str): The file path inside the bucket.

            context (grpc.ServicerContext): The gRPC context for handling the request.

        Returns:
            pb2.LoadFromGcsResponse: A response containing:
                - message (str): A success message indicating the collection has been loaded.
        """
        collection_name = request.collection_name
        creds = json.loads(request.creds)
        self.cm.set_collection_write_mode(collection_name, in_write_mode=True)
        try:
            Collection.load_from_gcs(collection_name, creds, request.bucket, request.path)
        finally:
            # Always set back to read mode, even if an error occurs
            self.cm.set_collection_write_mode(collection_name, in_write_mode=False)

        self.logger.debug(self._make_log(request.client_id, request.collection_name))

        return pb2.LoadFromGcsResponse(
            message=(f"Collection '{request.collection_name}' successfully loaded from"
                     f"{request.bucket}/{request.path}"))

    def start_idle_check(self):
        """
        Starts the idle check mechanism for monitoring inactive clients.
        This method triggers the idle check process within the connection manager
        to track client activity and enforce timeout policies.
        """
        self.cm.start_idle_check()

    def stop_idle_check(self):
        """
        Stops the idle check mechanism.
        This method disables the idle check process within the connection manager,
        preventing automatic disconnection of inactive clients as part of the graceful shut-down.
        """
        self.cm.stop_idle_check()

    @staticmethod
    def _get_index_config_request(index_config_request):
        return {
            item.key: from_flex_item(item.value)
            for item in index_config_request.items
        }

    @staticmethod
    def _get_schema_request(schema_request):
        return [
            (schema_item.field_name, schema_item.field_type, schema_item.is_nullable)
            for schema_item in schema_request
        ]

    @staticmethod
    def _get_metadata_or_filter_request(flex_map_list):
        return {
            flex_map.key: {item.key: from_flex_item(item.value) for item in flex_map.items}
            for flex_map in flex_map_list
        }

    def _reset_idle_timer(self, client_id):
        """
        Resets the idle timer for a specific client.

        Args:
            client_id (str): The client ID whose idle timer should be reset.
        """
        self.cm.reset_idle_timer(client_id)

    def _validate_client(self, client_id):
        """
        Validates whether a given client ID corresponds to an active client.

        Args:
            client_id (str): The client ID to validate.

        Raises:
            InvalidRequest: If the client ID is None.
            ConnectionTimedOutError: If the client has exceeded the idle timeout.
        """
        if client_id is None:
            raise InvalidRequest("Could not find the connected client")
        client_exists = self.cm.check_client_exists(client_id)
        if not client_exists:
            raise ConnectionTimedOutError(f"Idle timeout of {self.idle_timeout} seconds exceeded")

    def _validate_collection(self, client_id=None, collection_name=None):
        """
        Validates whether a specified collection exists.
        If a client ID is provided, the corresponding collection name is retrieved.

        Args:
            client_id (str, optional): The client ID to whose collection to be validated.
            collection_name (str, optional): The name of the collection to validate.

        Raises:
            InvalidRequest: If the collection name is None.
            CollectionNotExistsError: If the collection does not exist.
        """
        if client_id is not None:
            collection_name = self.cm.get_collection_name(client_id)
        if collection_name is None:
            raise InvalidRequest("Could not find the collection 'None'")
        if not Collection.exists(collection_name):
            raise CollectionNotExistsError(f"Collection '{collection_name}' does not exist")

    def _make_log(self, client_id, collection_name, step="result", extra={}):
        msg = {
            "client_id": client_id,
            "collection": collection_name,
            "step": step,
            "extra": extra
        }
        return msg


# function to start the server
def serve(host, port, service_account_info, server_config_path):

    setup_server_logging()

    # Create and configure the gRPC server
    config = load_yaml(server_config_path)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config["max_workers"]),
        options=[tuple(extra) for extra in config["grpc_options"]])
    service = MellowServer()
    pb2_grpc.add_MellowServiceServicer_to_server(service, server)

    # Bind the server to the specified host and port
    credentials = get_server_credentials(service_account_info)
    server.add_secure_port(f"{host}:{port}", credentials)

    # Start idle check in a separate thread
    service.start_idle_check()

    # Define the shutdown handler
    def shutdown_gracefully(signal, frame):
        print("Shutting down...")
        service.stop_idle_check()  # Stop the idle check thread
        # Gracefully stop the server, wait up to x seconds for active requests to finish
        server.stop(config["wait_before_shutdown"])
        sys.exit(0)

    # Listen for termination signals (Ctrl+C or SIGTERM)
    signal.signal(signal.SIGINT, shutdown_gracefully)  # Handle SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, shutdown_gracefully)  # Handle SIGTERM (Graceful termination)

    server.start()
    print(f"Server started on host={host} port={port}")

    server.wait_for_termination()
