class Error(Exception):
    pass


# one error to catch everything which I assume is not the best idea in real application
# please forgive me
class AnyError(Error):
    def __init__(self, message="Any Error! Jesteśmy zgubieni!") -> None:
        self.message = message
        super().__init__(self.message)


class MyError(Error):
    def __init__(self, message="Treść!") -> None:
        self.message = message
        super().__init__(self.message)