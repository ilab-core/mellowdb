import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))

from mellow_db.server import serve
from dotenv import load_dotenv
import json

load_dotenv(override=True)


if __name__ == "__main__":
    try:
        port = os.getenv("MELLOW_PORT")
        service_account_info = json.loads(os.getenv("GCP_SERVICE_ACCOUNT"))
        serve("0.0.0.0", port, service_account_info, "config/server.config.yaml")
    except Exception as e:
        print(f"Server failed to start: {e}")
