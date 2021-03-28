class Error(Exception):
    pass


class AnyError(Error):
    def __init__(self, message="Any Error! Jesteśmy zgubieni!") -> None:
        self.message = message
        super().__init__(self.message)