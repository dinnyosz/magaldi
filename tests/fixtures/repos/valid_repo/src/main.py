"""Main module for testing."""


def hello():
    """Say hello."""
    print("Hello, World!")


class Greeter:
    """A greeter class."""

    def __init__(self, name: str):
        self.name = name

    def greet(self):
        """Greet someone."""
        return f"Hello, {self.name}!"
