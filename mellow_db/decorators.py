import builtins
import traceback
from functools import wraps

import grpc

from mellow_db import exceptions


def with_rollback(func):
    """
    Decorator to handle session rollback and exceptions for db operations.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.rollback()
            print(f"Rolled-back due to error in {func.__name__}: {e}")
            raise e
    return wrapper


def request_handler(response, client_id_check=True, collection_name_check=True, reset_idle_timer=True):
    """
    Decorator to handle request validation and error handling for gRPC requests.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, request, context):
            try:
                _log_request_handler(self, request, "start")
                if client_id_check:
                    _log_request_handler(self, request, "validate_client")
                    self._validate_client(request.client_id)
                if collection_name_check:
                    _log_request_handler(self, request, "validate_collection")
                    self._validate_collection(client_id=request.client_id)
                # Reset the idle timer for each request
                if reset_idle_timer:
                    _log_request_handler(self, request, "reset_timer")
                    self._reset_idle_timer(request.client_id)
                return func(self, request, context)
            except exceptions.ConnectionTimedOutError as e:
                traceback_info = traceback.format_exc()
                context.set_code(grpc.StatusCode.ABORTED)
                error = e
            except Exception as e:
                traceback_info = traceback.format_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                error = e
            error_type = f"{type(error).__name__}"
            context.set_details(str(error))
            context.set_trailing_metadata([("error_type", error_type)])
            _log_request_handler(self, request, "error",
                                 {"error_type": error_type,
                                  "error": str(error),
                                  "traceback": traceback_info})
            return response()
        return wrapper
    return decorator


def response_handler(collection_name_check=True):
    """
    Decorator to handle response validation and error handling for gRPC responses.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                if not self.client_id:
                    raise exceptions.ConnectionFailedError("Could not find an active connection")
                if collection_name_check and not self.collection_name:
                    raise exceptions.InvalidRequest(
                        "Could not find a connected collection. Call use_collection() first")
                return func(self, *args, **kwargs)
            except grpc.RpcError as rpc_error:
                status_code = rpc_error.code()
                details = rpc_error.details()
                if status_code in (grpc.StatusCode.ABORTED, grpc.StatusCode.INTERNAL):
                    error_type = rpc_error.trailing_metadata()[0].value
                    if hasattr(exceptions, error_type):
                        error_class = getattr(exceptions, error_type)
                    elif hasattr(builtins, error_type):
                        error_class = getattr(builtins, error_type)
                    else:
                        error_class = Exception
                        details = (f"Unknown exception type '{error_type}' with status code '{status_code}'."
                                   f"Error details: {details}")
                    raise error_class(details)
                else:
                    raise Exception(f"Unhandled gRPC error [{status_code}]: {details}")
        return wrapper
    return decorator


def _log_request_handler(instance, request, step, extra={}):
    instance.logger.debug(instance._make_log(
        getattr(request, "client_id", None),
        getattr(request, "collection_name", None),
        step,
        extra))
