import threading
import time
from collections import defaultdict

from mellow_db.collection import Collection
from mellow_db.exceptions import ConcurrentWriteError


class ConnectionManager():
    """
    Manages client connections to collections, ensuring proper handling of idle clients,
    concurrent writes, and memory usage.

    Attributes:
        col_name_to_collection (dict): Maps collection names to Collection objects.
        col_name_to_client (defaultdict): Tracks connected clients for each collection.
        client_to_info (dict): Stores client IDs, their assigned collection, and last active time.
        collections_in_write (set): Keeps track of collections currently in write mode.
        idle_timeout (int): Maximum idle time (in seconds) before a client is disconnected.
        idle_check_frequency (int): Frequency (in seconds) for checking idle clients.
        idle_check_thread (Thread or None): Thread for managing idle clients.
        idle_check_running (bool): Flag to indicate if idle check is running.
        idle_check_event (Event): Event to manage idle check thread.
        lock (Condition): Lock for thread-safe operations.
    """

    def __init__(self, idle_timeout):
        """
        Initializes the ConnectionManager with a specified idle timeout.

        Args:
            idle_timeout (int): Time in seconds before an idle client is removed.
        """
        self.col_name_to_collection = {}
        self.col_name_to_client = defaultdict(list)
        self.client_to_info = {}
        self.collections_in_write = set()

        self.idle_timeout = idle_timeout
        self.idle_check_frequency = self.idle_timeout // 2
        self.idle_check_thread = None
        self.idle_check_running = False
        self.idle_check_event = threading.Event()
        self.lock = threading.Condition()

    def start_idle_check(self):
        """Starts a background thread to monitor and disconnect idle clients."""
        self.idle_check_running = True
        self.idle_check_event.clear()
        self.idle_check_thread = threading.Thread(target=self._manage_idle_clients, daemon=True)
        self.idle_check_thread.start()

    def stop_idle_check(self):
        """Stops the idle check thread gracefully."""
        self.idle_check_running = False
        self.idle_check_event.set()  # Wake up the thread if it is sleeping
        if self.idle_check_thread is not None:
            self.idle_check_thread.join(timeout=10)
            if self.idle_check_thread.is_alive():
                print("Idle check thread did not terminate in time. Force termination manually")
            self.idle_check_thread = None

    def add_client(self, client_id):
        """Adds a new client without assigning a collection yet.

        Args:
            client_id (str): Unique identifier for the client.
        """
        with self.lock:
            self.client_to_info[client_id] = (None, time.time())

    def assign_collection(self, client_id, collection_name):
        """Assigns a client to a collection and updates last activity time.

        Args:
            client_id (str): Unique identifier for the client.
            collection_name (str): Name of the collection to assign.
        """
        with self.lock:
            self.client_to_info[client_id] = (collection_name, time.time())
            self.col_name_to_client[collection_name].append(client_id)

    def get_clients_of_collection(self, collection_name, use_lock=True):
        """Returns a list of clients connected to a specific collection.

        Args:
            collection_name (str): The collection name.
            use_lock (bool): Whether to use locking for thread safety.

        Returns:
            list: List of client IDs connected to the collection.
        """
        if use_lock:
            self.lock.acquire()
        try:
            return self.col_name_to_client.get(collection_name, [])
        finally:
            if use_lock:
                self.lock.release()

    def check_client_exists(self, client_id):
        """Checks if a client exists in the system.

        Args:
            client_id (str): Unique identifier for the client.

        Returns:
            bool: True if the client exists, False otherwise.
        """
        with self.lock:
            return client_id in self.client_to_info

    def get_idle_clients(self):
        """Finds clients that have been idle past the timeout period.

        Returns:
            list: List of idle client IDs.
        """
        current_time = time.time()
        with self.lock:
            idle_clients = [
                client_id
                for client_id, (_, last_activity_time) in self.client_to_info.items()
                if current_time - last_activity_time >= self.idle_timeout
            ]
            return idle_clients

    def reset_idle_timer(self, client_id):
        """Resets the idle timer for a client.

        Args:
            client_id (str): Unique identifier for the client.
        """
        with self.lock:
            collection_name = self.get_collection_name(client_id, use_lock=False)
            self.client_to_info[client_id] = (collection_name, time.time())

    def remove_client(self, client_id, use_lock=True):
        """Removes a client and cleans up collection references.

        Args:
            client_id (str): Unique identifier for the client.
            use_lock (bool): Whether to use locking for thread safety.

        Returns:
            str or None: The collection name the client was connected to, or None.
        """
        if use_lock:
            self.lock.acquire()
        try:
            collection_name = self.get_collection_name(client_id, use_lock=False)
            # client is not already disconnected from collection
            if collection_name is not None:
                # remove client from collection's clients
                self.col_name_to_client[collection_name].remove(client_id)
                # if there is not any left client using the collection, remove to free up the memory
                if not self.get_clients_of_collection(collection_name, use_lock=False):
                    del self.col_name_to_collection[collection_name]
            # remove client itself
            del self.client_to_info[client_id]
            return collection_name
        finally:
            if use_lock:
                self.lock.release()

    def load_collection_to_memory(self, collection_name):
        """Loads a collection into memory if not already loaded.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Collection: The loaded collection object.
        """
        with self.lock:
            if self.get_clients_of_collection(collection_name, use_lock=False):
                return self.col_name_to_collection[collection_name]  # already loaded in memory
        collection_ = Collection(name=collection_name)
        with self.lock:
            self.col_name_to_collection[collection_name] = collection_
        return collection_

    def get_collection(self, client_id):
        """
        Retrieve the collection that the client assigned to.

        Args:
            client_id (str or int): The client ID for which to retrieve the collection.

        Returns:
            object: The collection object that the client assigned to.
        """
        with self.lock:
            collection_name = self.get_collection_name(client_id, use_lock=False)
            return self.col_name_to_collection[collection_name]

    def get_collection_name(self, client_id, use_lock=True):
        """
        Retrieve the collection name that the client assigned to.

        Args:
            client_id (str or int): The client ID for which to retrieve the collection name.
            use_lock (bool, optional): Whether to acquire a lock before accessing the mapping.
                Defaults to True.

        Returns:
            str: The name of the collection object that the client assigned to.
        """
        if use_lock:
            self.lock.acquire()  # Manually acquire the lock
        try:
            return self.client_to_info[client_id][0]
        finally:
            if use_lock:
                self.lock.release()

    def set_collection_write_mode(self, collection_name, in_write_mode):
        """Manages write access to a collection to prevent concurrent writes.

        Args:
            collection_name (str): Name of the collection.
            in_write_mode (bool): Whether to enable write mode.

        Raises:
            ConcurrentWriteError: If another write is already in progress.
        """
        with self.lock:
            if in_write_mode:
                # TODO There can be a queue mechanism to handle multiple write requests
                # For now, only one write request is allowed at a time
                if collection_name in self.collections_in_write:
                    raise ConcurrentWriteError(
                        f"Detected concurrent write request to collection '{collection_name}'"
                    )
                # Block new reads until the write is complete
                self.collections_in_write.add(collection_name)
            else:
                self.collections_in_write.discard(collection_name)
                self.lock.notify_all()  # Notify waiting readers

    def wait_until_collection_available(self, collection_name):
        """
        Wait until the specified collection is available for access.

        Args:
            collection_name (str): The name of the collection to wait for.
        """
        # Wait if the collection is in write mode
        with self.lock:
            while collection_name in self.collections_in_write:
                self.lock.wait()  # Wait until the write mode is cleared

    def _manage_idle_clients(self):
        """Periodically checks for idle clients and removes them."""
        while self.idle_check_running:
            idle_clients = self.get_idle_clients()
            with self.lock:
                for client_id in idle_clients:
                    self.remove_client(client_id, use_lock=False)
                    print(f"Client {client_id} disconnected due to idle timeout")
            # Wait for the next idle check or an external wake-up
            self.idle_check_event.wait(self.idle_check_frequency)
            self.idle_check_event.clear()
