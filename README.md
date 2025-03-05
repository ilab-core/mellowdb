# MellowDB

MellowDB is a lightweight vector database designed for efficient similarity search with optional metadata handling. It supports multiple concurrent requests, manages multiple collections, and is suited for frequent querying with minimal updates.

### Core Components

- **Vector Indexing**: Utilizes [FAISS](https://faiss.ai/) (Facebook AI Similarity Search) for scalable and fast vector indexing.
- **Metadata Storage**: Integrates [SQLite](https://www.sqlite.org/) with [SQLAlchemy](https://www.sqlalchemy.org/) as the ORM layer for structured data management.
- **Communication**: Uses [gRPC](https://grpc.io/) with [Protocol Buffers](https://developers.google.com/protocol-buffers) (proto3) for efficient remote procedure calls.

MellowDB is designed to provide a simple and effective solution for applications requiring vector search with metadata support.


## How To Use?

The documentation is currently being updated. For more details, please refer to the [docs](docs).

## License

MellowDB is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
