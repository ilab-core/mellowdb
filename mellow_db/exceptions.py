class ConcurrentWriteError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CollectionExistsError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CollectionNotExistsError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MissingInfoInCollection(Exception):
    def __init__(self, message):
        super().__init__(message)


class CollectionHasActiveClientsError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ConnectionFailedError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ConnectionTimedOutError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ActiveConnectionFound(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidRequest(Exception):
    def __init__(self, message):
        super().__init__(message)


class ResourceNotFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)
