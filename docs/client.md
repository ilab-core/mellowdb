# MellowClient

MellowClient is a gRPC-based client for interacting with MellowDB. This client allows users to establish secure connections, create collections, manage data, and efficiently retrieve and search within the stored items.

## Initialization

`__init__(host, port, service_account_info, connect_timeout=90, channel_options=[...])`

#### Description
Initializes the `MellowClient` instance and establishes a secure connection with MellowDB.

#### Parameters
- **host** (`str`): The Mellow service host address.
- **port** (`int`): The port number to connect to.
- **service_account_info** (`dict`): GCS authentication account credentials to load service certificates for MellowDB.
- **connect_timeout** (`int`, optional): Connection timeout in seconds (default: `90s`).
- **channel_options** (`list`, optional): Additional gRPC options for communication.

#### Raises
- **ConnectionFailedError**: If the connection to the Mellow service fails.

---

## Collection Management

#### `create_collection(collection_name, index_config={}, collection_schema=[])`

#### Description
Creates a new collection in the Mellow service.

#### Parameters
- **collection_name** (`str`): The name of the collection.
- **index_config** (`dict`, optional): Configuration for FAISS indexing (default: `{}`).
- **collection_schema** (`list`, optional): Schema definition for metadata (default: `[]`).

#### Returns
- **`str`**: Response message from the Mellow service.

---

#### `use_collection(collection_name)`

#### Description
Sets the currently active collection for performing operations.

#### Parameters
- **collection_name** (`str`): Name of the collection to activate.

#### Returns
- **`str`**: Response message from the Mellow service.

#### Raises
- **ActiveConnectionFound**: If a collection is already in use.

---

#### `delete_collection(collection_name)`

#### Description
Deletes a specified collection from the server.

#### Parameters
- **collection_name** (`str`): Name of the collection to delete.

#### Returns
- **`str`**: A success message confirming deletion.

---

#### `get_collection_item_count()`

#### Description
Fetches the total number of items in the currently active collection.

#### Returns
- **`item_count`** (`int`): The total number of items in the collection.

---

#### `get_collection_info()`

#### Description
Provides detailed metadata about the currently active collection.

#### Returns
- **`collection_info`** (`dict`): A dictionary containing:
  - **name** (`str`): The name of the collection.
  - **item_count** (`int`): The number of items in the collection.
  - **size_in_bytes** (`int`): The total size of the collection in bytes.
  - **faiss_index_type** (`str`): The FAISS index type used.
  - **faiss_index_metric** (`str`): The FAISS index metric used.
  - **embedding_dim** (`int`): The dimensionality of the embeddings.
  - **primary_keys** (`list`): List of primary key field names.
  - **meta_columns** (`list`): A list of metadata columns in the collection, with details:
    - **name** (`str`): The column name.
    - **type** (`str`): The data type.
    - **is_nullable** (`bool`): Whether the column allows null values.
    - **is_index** (`bool`): Whether the column is indexed.
    - **default** (`any`): The default value for the column.

---

#### `eval(subset_size, test_size, k=30, tolerance=1e-5)`

#### Description
Evaluate the performance of the current collection using a subset of items. This method evaluates the collection's performance by testing it on a subset of items and calculating performance metrics such as precision, recall, and search time.

#### Parameters
- **subset_size** (`int`): The number of items to include in the evaluation subset.
- **test_size** (`int`): The number of test items to evaluate.
- **k** (`int`, optional): The number of nearest neighbors to consider (default: `30`).
- **tolerance** (`float`, optional): The tolerance level for evaluation (default: `1e-5`).

#### Returns
- **`tuple`**: A tuple containing the following performance metrics:
  - `avg_mellow_time` (`float`): Average search time using MellowDB.
  - `avg_cosine_time` (`float`): Average search time using scikit-learn's cosine similarity.
  - `avg_precision` (`float`): Average precision comparing MellowDB and scikit-learn results.
  - `avg_recall` (`float`): Average recall comparing MellowDB and scikit-learn results.
  - `sum_diffs` (`float`): Number of similarity score differences between MellowDB and scikit-learn results.
  - `avg_diffs` (`float`): Average number of similarity score differences between MellowDB and scikit-learn results.

---

## Data Management

#### `add(key_to_data=None, key_to_metadata=None, upsert=False)`

#### Description
Adds or updates items in the currently active collection.

#### Parameters
- **key_to_data** (`dict`, optional): A mapping of keys to embedding data.
- **key_to_metadata** (`dict`, optional): A mapping of keys to metadata.
- **upsert** (`bool`, optional): Whether to update existing items (default: `False`).

#### Returns
- **`int`**: The updated item count in the collection.

#### Raises
- **ValueError**: If neither `key_to_data` nor `key_to_metadata` is provided.
- **ValueError**: If the lengths or keys of `key_to_data` and `key_to_metadata` don't match.

---

#### `get(where=None, projection=None, limit=None)`

#### Description
Retrieves items from the currently active collection based on filters.

#### Parameters
- **where** (`dict`, optional): Filtering criteria. Uses MongoDB-like syntax, e.g., `{"field_name": {"$eq": value}}`.
- **projection** (`list`, optional): List of fields to return (default: only key).
- **limit** (`int`, optional): Maximum number of results (default: no limit).

#### Returns
- **`list`**: A list of tuples representing the retrieved items.

#### Raises
- **ValueError**: If `projection` is an empty list.
- **ValueError**: If `limit` is set to 0.

---

#### `search(query_keys, where=None, projection=None, n_results=None, threshold=None, not_exists_ok=False)`

#### Description
Search for items in the collection based on query keys and optional filters.

#### Parameters
- **query_keys** (`list`): A list of keys to search for.
- **where** (`dict`, optional): A dictionary to filter search results (e.g., `{"field_name": {"$eq": value}}`).
- **projection** (`list`, optional): Fields to return (default: `["key", "similarity"]`).
- **n_results** (`int`, optional): Maximum number of results (must be > 0).
- **threshold** (`float`, optional): Minimum similarity threshold for filtering.
- **not_exists_ok** (`bool`, optional): Whether to allow searches for non-existent items.

#### Returns
- **`list`**: A list of tuples containing the requested fields for each search result.

---

#### `disconnect()`

#### Description
Closes the connection and disconnects the client.

#### Returns
- **`str`**: A message confirming the disconnection.

---

## Data Backup and Restoration

#### `back_up(backup_dir)`

#### Description
Creates a backup of the currently active collection and stores it in a specified local (remote) directory.

#### Parameters
- **backup_dir** (`str`): The directory where the backup should be stored.

#### Returns
- **message** (`str`): A response message indicating the status of the backup.
- **backup_full_path** (`str`): The full path to the created backup file.

---

#### `back_up_to_gcs(creds, bucket, backup_dir)`

#### Description
Creates a backup of the currently active collection and uploads it to a specified Google Cloud Storage (GCS) bucket.

#### Parameters
- **creds** (`dict` or `str`): Google Cloud credentials as a dictionary or JSON string.
- **bucket** (`str`): The name of the GCS bucket where the backup will be stored.
- **backup_dir** (`str`): The directory path within the bucket where the backup should be saved.

#### Returns
- **message** (`str`): A response message indicating the status of the backup.
- **backup_full_path** (`str`): The full path to the backup file in GCS.

---

#### `load_from_path(path, collection_name)`

#### Description
Restores a collection from a backup file stored on the local file system.

#### Parameters
- **path** (`str`): The file path to the backup.
- **collection_name** (`str`): The name of the collection to be restored.

#### Returns
- **message** (`str`): A response message indicating the status of the process.

---

#### `load_from_gcs(creds, bucket, path, collection_name)`

#### Description
Restores a collection from a backup file stored in a Google Cloud Storage (GCS) bucket.

#### Parameters
- **creds** (`dict` or `str`): Google Cloud credentials as a dictionary or JSON string.
- **bucket** (`str`): The name of the GCS bucket where the backup is stored.
- **path** (`str`): The file path to the backup within the bucket.
- **collection_name** (`str`): The name of the collection to be restored.

#### Returns
- **message** (`str`): A response message indicating the status of the process.
