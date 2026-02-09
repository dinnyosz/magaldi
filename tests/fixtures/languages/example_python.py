"""Example Python file exercising all extractable element types.

This fixture ensures the magaldi index always contains Python-specific
elements (class, enum, decorator, async, nested) when parsing its own repo.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0
SUPPORTED_FORMATS = ("json", "yaml", "toml")

# Variables
_registry: dict[str, type] = {}
_initialized = False


class Color(Enum):
    """Example enum with auto values."""

    RED = auto()
    GREEN = auto()
    BLUE = auto()


class Priority(Enum):
    """Example enum with explicit values."""

    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 100


class Serializable(Protocol):
    """Example protocol (interface-like) for structural typing."""

    def serialize(self) -> bytes: ...
    def deserialize(self, data: bytes) -> None: ...


@dataclass
class Config:
    """Example dataclass with fields and methods."""

    name: str
    timeout: float = DEFAULT_TIMEOUT
    retries: int = MAX_RETRIES
    tags: list[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate configuration values."""
        return self.timeout > 0 and self.retries >= 0

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {"name": self.name, "timeout": self.timeout, "retries": self.retries}


class BaseProcessor:
    """Base class with common processing logic."""

    def __init__(self, config: Config):
        self.config = config
        self._cache: dict[str, str] = {}

    def process(self, data: str) -> str:
        """Process input data."""
        raise NotImplementedError

    def _validate_input(self, data: str) -> bool:
        """Validate input before processing."""
        return bool(data and len(data) > 0)

    @staticmethod
    def create_default() -> "BaseProcessor":
        """Factory method to create default processor."""
        return BaseProcessor(Config(name="default"))


class TextProcessor(BaseProcessor):
    """Processor for text data with caching."""

    def process(self, data: str) -> str:
        """Process text by normalizing whitespace."""
        if data in self._cache:
            return self._cache[data]
        result = " ".join(data.split())
        self._cache[data] = result
        return result

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()


def parse_config(path: Path) -> Config:
    """Parse configuration from a file path."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return Config(name=path.stem)


def register_processor(name: str, processor_class: type) -> None:
    """Register a processor class by name."""
    _registry[name] = processor_class


async def fetch_remote_config(url: str, timeout: float = 10.0) -> Config:
    """Async function to fetch configuration from a remote URL."""
    logger.info("Fetching config from %s", url)
    return Config(name="remote", timeout=timeout)


async def process_batch(items: list[str], processor: BaseProcessor) -> list[str]:
    """Async batch processing of multiple items."""
    return [processor.process(item) for item in items]


def with_retry(func):
    """Decorator that adds retry logic."""

    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
        return None

    return wrapper


@with_retry
def unreliable_operation(data: str) -> str:
    """An operation that might fail and needs retries."""
    return data.upper()
