import logging
import os
import shutil

from sqlalchemy import (JSON, Column, Float, Integer, String, Table, and_,
                        create_engine, inspect, or_)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from mellow_db.decorators import with_rollback
from mellow_db.exceptions import ResourceNotFoundError
from mellow_db.storage import (download_from_gcs, upload_to_gcs,
                               validate_folder, validate_gcs_folder)

operator_map = {
    "$eq": lambda col, val: col == val,  # Equal
    # Not equal, including null handling
    "$ne": lambda col, val: or_(col != val, col.is_(None)) if val is not None else col.isnot(None),
    "$gt": lambda col, val: col > val,  # Greater than
    "$lt": lambda col, val: col < val,  # Less than
    "$gte": lambda col, val: col >= val,  # Greater than or equal
    "$lte": lambda col, val: col <= val,  # Less than or equal
    # In list, including null handling
    "$in": lambda col, val: or_(col.is_(None), col.in_(val)) if None in val else col.in_(val),
    # defining $nin with null values is difficult with sqlite, will be added upon request
    # $like, $ilike, $exists, will be added upon request
}


class MetadataDB:
    """
    A class for managing an SQLite database with metadata storage.

    Attributes:
        db_url (str): The database connection URL.
        db_path (str): The local path of the SQLite database.
        Base (declarative_base): SQLAlchemy base class for table definitions.
        engine (Engine): SQLAlchemy engine for database connection.
        Session (sessionmaker): A configured session factory.
        session (Session): An active database session.
        table_name (str): The name of the metadata table.
        table_ (Table): The metadata table object.
        logger (Logger): Logger for logging database operations.
    """

    def __init__(self, db_url, schema=None, echo=False):
        """
        Initialize the MetadataDB instance.

        Args:
            db_url (str): The SQLite database URL.
            schema (dict, optional): Schema definition for the metadata table.
            echo (bool, optional): If True, enables SQLAlchemy query logging.

        Raises:
            ValueError: If schema is not provided when creating a new database.
        """
        self.logger = logging.getLogger("metadata_db")

        # Initialize the database connection, session, and tables
        self.db_url = db_url
        self.db_path = self.db_url.replace("sqlite:///", "")
        self.Base = declarative_base()
        self.engine = create_engine(db_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        # Generate and create tables
        self.table_name = "metadata_table"
        # Check if the table already exists
        if self._table_exists():
            self.logger.info(self._make_log({
                "msg": "Table exists. Loading existing table",
                "table": self.table_name
            }))
            self.table_ = self._load_existing_table()
        else:
            self.logger.info(self._make_log({
                "msg": "Table does not exist. Creating new table",
                "table": self.table_name
            }))
            if schema is not None:
                self._validate_schema(schema)
                self.table_ = self._setup_metadata_table(schema)
                self.Base.metadata.create_all(self.engine)
            else:
                raise ValueError("Schema is required for the first initialization")

        self.logger.info(self._make_log({
            "msg": "SqliteDB ready",
            "num_items": self.get_count()
        }))

    @classmethod
    def load_from_data(cls, data, db_url, schema, upsert=False):
        """
        Load metadata from a data dict into a new SQLite database instance.

        Args:
            data (dict): The data to insert into the database.
            db_url (str): The SQLite database URL.
            schema (dict): Schema definition for the metadata table.
            upsert (bool, optional): If True, updates existing records.

        Returns:
            MetadataDB: A new instance of the MetadataDB class.
        """
        instance = cls(db_url, schema)
        instance.insert(data, upsert=upsert).commit()

        instance.logger.info(instance._make_log({
            "msg": "SQLiteDB successfully loaded",
            "num_items": instance.get_count()
        }))
        return instance

    @classmethod
    def load_from_path(cls, file_path, db_url, schema):
        """
        Load an existing SQLite database from a local file.

        Args:
            file_path (str): The path to the SQLite database file.
            db_url (str): The SQLite database URL.
            schema (dict): Schema definition for the metadata table.

        Returns:
            MetadataDB: A new instance of the MetadataDB class.
        """
        shutil.copy(file_path, db_url.replace("sqlite:///", ""))
        instance = cls(db_url=db_url, schema=schema)

        instance.logger.info(instance._make_log({
            "msg": "SQLiteDB successfully loaded",
            "num_items": instance.get_count(),
            "data_path": file_path
        }))
        return instance

    @classmethod
    def load_from_gcs(cls, creds, bucket, file_path, db_url, schema):
        """
        Load an existing SQLite database from Google Cloud Storage.

        Args:
            creds (dict): Credentials for accessing Google Cloud Storage.
            bucket (str): The GCS bucket name.
            file_path (str): The path to the database file in the bucket.
            db_url (str): The SQLite database URL.
            schema (dict): Schema definition for the metadata table.

        Returns:
            MetadataDB: A new instance of the MetadataDB class.
        """
        destination_folder = os.path.dirname(db_url.replace("sqlite:///", ""))
        download_from_gcs(creds, bucket, file_path, destination_folder)
        instance = cls(db_url=db_url, schema=schema)

        instance.logger.info(instance._make_log({
            "msg": "SQLiteDB successfully loaded",
            "num_items": instance.get_count(),
            "data_path": f'gs:/{bucket}/{file_path}'
        }))
        return instance

    def get_count(self):
        """
        Get the number of rows in the metadata table.

        Returns:
            int: The number of rows in the table.
        """
        return (self.session.query(self.table_).count()
                if self.table_name in inspect(self.engine).get_table_names()
                else 0)

    def get_file_size(self):
        """
        Get the file size of the SQLite database.

        Returns:
            int: The file size in bytes.
        """
        return os.path.getsize(self.db_path)

    def get_info(self):
        """
        Retrieve metadata information about the database.

        Returns:
            dict: A dictionary containing:
                - "primary_keys" (list): Names of the primary key columns.
                - "meta_columns" (list of dicts): Metadata about each column, including:
                    - "name" (str): Column name.
                    - "type" (str): Column data type.
                    - "is_nullable" (bool): Whether the column allows NULL values.
                    - "is_index" (bool): Whether the column has an index.
                    - "default" (any): Default value for the column, if any.
                - "item_count" (int): Total number of records in the table.
                - "size_in_bytes" (int): Size of the SQLite database file in bytes.
        """
        meta_info_obj = self.Base.metadata.tables.get(self.table_name)
        info = {
            "primary_keys": [str(column.name)
                             for column in meta_info_obj.primary_key.columns
                             ],
            "meta_columns": [{
                "name": str(column.name),
                "type": str(column.type),
                "is_nullable": column.nullable,
                "is_index": column.index is not None,
                "default": column.default,
            } for column in meta_info_obj.columns
            ],
            "item_count": self.get_count(),
            "size_in_bytes": self.get_file_size(),
        }
        return info

    @with_rollback
    def insert(self, data_dict, upsert=False):
        """
        Insert or update multiple records in the database.

        Args:
            data_dict (dict): A dictionary where keys are primary keys and values
                are dictionaries containing column-value pairs.
            upsert (bool, optional): If True, updates existing records. If False,
                raises an error if a primary key conflict occurs.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ValueError: If upsert is False and any of the primary keys already exist.
        """
        primary_keys = list(data_dict.keys())
        existing_records = (
            self.session.query(self.table_)
            .filter(self.table_.key.in_(primary_keys))
            .all()
        )
        existing_record_keys = {record.key for record in existing_records}
        update_mappings = [{"key": record.key, **data_dict[record.key]}
                           for record in existing_records]
        insert_mappings = [{"key": key, **data_dict[key]}
                           for key in data_dict if key not in existing_record_keys]
        # Perform the bulk update
        if upsert and update_mappings:
            self.session.bulk_update_mappings(self.table_, update_mappings)
        elif not upsert and update_mappings:
            raise ValueError(
                f"Primary key must be unique. {len(existing_records)} records already exists")
        # Perform the bulk insert
        if insert_mappings:
            self.session.bulk_insert_mappings(self.table_, insert_mappings)
        return self

    @with_rollback
    def update(self, primary_key, update_dict):
        """
        Update an existing record in the database by its primary key.

        Args:
            primary_key (any): The primary key of the record to be updated.
            update_dict (dict): Dictionary of column-value pairs to update.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ValueError: If the specified primary key does not exist in the database.
        """
        record = self.read(primary_key)
        if record:
            for key, value in update_dict.items():
                setattr(record, key, value)
        else:
            raise ValueError(f"Could not find primary key '{primary_key}'")
        return self

    def read(self, primary_key):
        """
        Retrieve a record from the database using its primary key.

        Args:
            primary_key (any): The primary key of the record to retrieve.

        Returns:
            sqlalchemy.orm.query.Query: The record object if found, otherwise None.
        """
        return self.session.query(self.table_).filter_by(key=primary_key).first()

    def query(self, where={}, projection=["key"], limit=None, return_field_names=False):
        """
        Query rows based on Mongo-like filter criteria.

        Args:
            where (dict): Mongo-like filter criteria as a dictionary. Example:
                        {"field_name": {"$eq": value}}.
            projection (list): List of column names to include in the result.
            limit (int, optional): Maximum number of rows to return. If None, no limit is applied.
            return_field_names (bool): If True, returns the results as a list of dictionaries with
                                    field names as keys. If False, returns results as tuples.

        Returns:
            list: A list of dictionaries or tuples representing the queried rows,
                depending on the value of `return_field_names`.

        Raises:
            ValueError: If an unsupported operator is used in the `where` filter criteria.
        """
        projection_columns = [(field, getattr(self.table_, field)) for field in projection]
        query = self.session.query(*(col for _, col in projection_columns))
        filter_criteria = []

        # convert from mongo-like filtering to SQL filtering
        for field, condition in where.items():
            column = getattr(self.table_, field)
            for operator_type, operator in condition.items():
                if operator_type in operator_map:
                    filter_criteria.append(operator_map[operator_type](column, operator))
                else:
                    raise ValueError(f"Unsupported operator '{operator_type}'")
        if filter_criteria:
            query = query.filter(and_(*filter_criteria))

        # apply the limit if specified
        if limit is not None:
            query = query.limit(limit)

        results = query.all()

        # pre-determine the output formatter with a lambda function
        if return_field_names:
            results_fn = lambda row: {field: getattr(row, field) for field, _ in projection_columns}
        else:
            results_fn = lambda row: tuple(getattr(row, field) for field, _ in projection_columns)
        results = [results_fn(row) for row in results]
        return results

    @with_rollback
    def delete(self, primary_key, not_exists_ok=False):
        """
        Delete a record from the database by its primary key.

        Args:
            primary_key (Any): The primary key of the record to be deleted.
            not_exists_ok (bool, optional): If True, suppresses errors when the record
                does not exist. Defaults to False.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ValueError: If the record does not exist and `not_exists_ok` is False.
        """
        record = self.read(primary_key)
        if record:
            self.session.delete(record)
        else:
            if not not_exists_ok:
                raise ValueError(f"Could not find primary key '{primary_key}'")
        return self

    def back_up(self, folder_path):
        """
        Create a local backup of the SQLite database.

        Args:
            folder_path (str): The destination folder where the database backup
                should be saved.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ValueError: If the folder path is invalid.
        """
        validate_folder(folder_path, not_empty_ok=True)
        shutil.copy2(self.db_path, folder_path)

        self.logger.info(self._make_log({
            "msg": "SQLiteDB successfully backed up",
            "num_items": self.get_count(),
            "backup_folder": folder_path}
        ))
        return self

    def back_up_to_gcs(self, creds, bucket, folder_path):
        """
        Upload a backup of the SQLite database to Google Cloud Storage (GCS).

        Args:
            creds (dict): Google Cloud credentials required for authentication.
            bucket (str): The name of the GCS bucket.
            folder_path (str): The destination folder in GCS where the database backup
                should be uploaded.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ValueError: If the folder path or bucket name is invalid.
        """
        validate_gcs_folder(creds, bucket, folder_path, not_empty_ok=True)
        upload_to_gcs(creds, self.db_path, bucket, folder_path)

        self.logger.info(self._make_log({
            "msg": "SQLiteDB successfully backed up",
            "num_items": self.get_count(),
            "backup_folder": f'gs:/{bucket}/{folder_path}'}
        ))
        return self

    def rollback(self):
        """Rollback uncommited changes."""
        self.session.rollback()

    def commit(self):
        """Commit pending changes to db."""
        self.session.commit()

    def close(self):
        """Close the database session."""
        self.session.close()

    def drop(self, not_exists_ok=False):
        """
        Drop the table from the database if it exists.
        Since MetadataDB contains only one table, calling this function effectively
        deletes the entire database.

        Args:
            not_exists_ok (bool, optional): If True, suppresses errors when the table
                does not exist. Defaults to False.

        Returns:
            MetadataDB: The current instance of MetadataDB.

        Raises:
            ResourceNotFoundError: If the table does not exist and `not_exists_ok` is False.
        """
        if self._table_exists():
            table_to_drop = self.Base.metadata.tables.get(self.table_name)
            if table_to_drop is not None:
                table_to_drop.drop(self.engine)
                self.logger.info(self._make_log({
                    "msg": "Dropped table",
                    "table": self.table_name}
                ))
            else:
                if not not_exists_ok:
                    raise ResourceNotFoundError(f"Table '{self.table_name}' is not defined in the SQLAlchemy metadata")
        else:
            if not not_exists_ok:
                raise ResourceNotFoundError(f"Table '{self.table_name}' does not exist in the database")
        return self

    def _table_exists(self):
        """
        Checks if the table specified by `self.table_name` exists in the database.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        return self.table_name in inspect(self.engine).get_table_names()

    def _validate_schema(self, schema):
        """
        Validates the schema provided for creating or modifying the table.

        Args:
            schema (list): A list of tuples where each tuple represents a field in the schema.
                Each tuple contains the field name (str), field type (str),
                and whether the field is nullable (bool).

        Raises:
            ValueError: If any of the following conditions are met:
                - The table already exists and a schema redefinition is attempted.
                - Reserved field names ('key', 'similarity', 'embedding') are included.
                - A nullable column has a type other than 'string' or 'list'.
        """
        if self._table_exists():
            raise ValueError("Table already exists but found re-definiton of schema")

        field_names = [schema_item[0] for schema_item in schema]
        field_types = [schema_item[1] for schema_item in schema]
        is_nullables = [schema_item[2] for schema_item in schema]
        if "key" in field_names:
            raise ValueError("Primary key field 'key' is auto-defined")
        if "similarity" in field_names:
            raise ValueError("Field name 'similarity' is reserved")
        if "embedding" in field_names:
            raise ValueError("Field name 'embedding' is reserved")

        for ix, type_ in enumerate(field_types):
            if type_ not in {"string", "list"} and is_nullables[ix]:
                raise ValueError((f"Error with field '{field_names[ix]}': nullable column is only allowed "
                                  f"with string or list type. Found '{field_types[ix]}'"))

    def _load_existing_table(self):
        """
        Loads an existing table dynamically based on the table name stored in `self.table_name`.

        Returns:
            class: A dynamically created model class based on the existing table structure.

        Raises:
            sqlalchemy.exc.NoSuchTableError: If the specified table does not exist.
        """
        existing_table = Table(self.table_name, self.Base.metadata, autoload_with=self.engine)
        # Create a model class dynamically based on the existing table
        model_class = type(self.table_name, (self.Base,), {'__table__': existing_table})
        return model_class

    def _setup_metadata_table(self, schema):
        """
        Generates a metadata table based on the provided schema.

        Args:
            schema (list): A list of tuples where each tuple represents a field in the schema.
                Each tuple contains the field name (str), field type (str),
                and whether the field is nullable (bool).

        Returns:
            class: A dynamically created model class for the metadata table.

        Raises:
            ValueError: If an unsupported field type is encountered in the schema.
        """
        columns = {
            "__tablename__": self.table_name,
            "__table_args__": {"extend_existing": True},
            "key": Column(String, primary_key=True)
        }
        for field_name, field_type, nullable in schema:
            if field_type == 'string':
                columns[field_name] = Column(String, nullable=nullable)
            elif field_type == 'float':
                columns[field_name] = Column(Float, nullable=nullable)
            elif field_type == 'integer':
                columns[field_name] = Column(Integer, nullable=nullable)
            elif field_type == 'list':
                columns[field_name] = Column(JSON, nullable=nullable)
            else:
                raise ValueError(f"Unsupported field type in metadata table: {field_type}")

        # Dynamically create the model class
        model_class = type(self.table_name, (self.Base,), columns)
        return model_class

    def _make_log(self, extra={}):
        msg = {
            "extra": {"db_url": self.db_url, **extra}
        }
        return msg
