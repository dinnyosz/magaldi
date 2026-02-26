"""Teatro Magaldi — Performer Management Module.

A 1930s Buenos Aires theater goes digital. This module handles the performers:
artists, roles, rehearsals, and the occasional diva tantrum.

Named after Agustín Magaldi, the legendary tango singer who (legend has it)
helped launch Eva Perón's career — and the namesake of this very project.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

stage_manager_log = logging.getLogger(__name__)

# Constants
MAX_ENCORES = 3
DEFAULT_REHEARSAL_HOURS = 30.0
SUPPORTED_ACT_TYPES = ("comedy", "drama", "musical")

# Variables
_performer_registry: dict[str, type] = {}
_auditions_open = False


class ActType(Enum):
    """The three pillars of Buenos Aires theater."""

    COMEDY = auto()
    DRAMA = auto()
    MUSICAL = auto()


class VocalRange(Enum):
    """Vocal classification for casting purposes."""

    SOPRANO = 1
    ALTO = 2
    TENOR = 3  # Agustín Magaldi was a legendary tenor
    BASS = 4


class Performable(Protocol):
    """Any entity that can take the stage."""

    def perform(self) -> bytes: ...
    def take_bow(self, data: bytes) -> None: ...


@dataclass
class PerformerProfile:
    """A performer's professional details for the teatro registry."""

    name: str
    rehearsal_hours: float = DEFAULT_REHEARSAL_HOURS
    encores: int = MAX_ENCORES
    roles: list[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Ensure the performer's profile is stage-ready."""
        return self.rehearsal_hours > 0 and self.encores >= 0

    def to_dict(self) -> dict:
        """Convert profile to a casting sheet."""
        return {"name": self.name, "rehearsal_hours": self.rehearsal_hours, "encores": self.encores}


class BasePerformer:
    """Every artist starts somewhere — this is where."""

    def __init__(self, profile: PerformerProfile):
        self.profile = profile
        self._costume_changes: dict[str, str] = {}

    def perform(self, script: str) -> str:
        """Deliver a performance. Subclasses bring the magic."""
        raise NotImplementedError

    def _check_stage_fright(self, script: str) -> bool:
        """Even legends get nervous."""
        return bool(script and len(script) > 0)

    @staticmethod
    def create_understudy() -> "BasePerformer":
        """Every show needs a backup plan."""
        return BasePerformer(PerformerProfile(name="understudy"))


class Diva(BasePerformer):
    """La Divina — she overrides everything, naturally."""

    def perform(self, script: str) -> str:
        """Deliver a performance with... artistic interpretation."""
        if script in self._costume_changes:
            return self._costume_changes[script]
        result = " ".join(script.split())  # She insists on perfect spacing
        self._costume_changes[script] = result
        return result

    def clear_wardrobe(self) -> None:
        """Between shows, La Divina demands a fresh wardrobe."""
        self._costume_changes.clear()


def parse_audition_results(path: Path) -> PerformerProfile:
    """Read audition results from the casting director's notes."""
    if not path.exists():
        raise FileNotFoundError(f"Audition results not found: {path}")
    return PerformerProfile(name=path.stem)


def register_performer(name: str, performer_class: type) -> None:
    """Add a performer to the teatro's roster."""
    _performer_registry[name] = performer_class


async def fetch_touring_schedule(url: str, timeout: float = 10.0) -> PerformerProfile:
    """Check if a touring performer is available for La Noche Estrellada."""
    stage_manager_log.info("Checking touring schedule at %s", url)
    return PerformerProfile(name="touring_artist", rehearsal_hours=timeout)


async def rehearse_ensemble(scripts: list[str], performer: BasePerformer) -> list[str]:
    """The whole troupe rehearses together. Chaos ensues."""
    return [performer.perform(script) for script in scripts]


def requires_rehearsal(func):
    """No one goes on stage without rehearsal. Not even La Divina."""

    def wrapper(*args, **kwargs):
        for attempt in range(MAX_ENCORES):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt == MAX_ENCORES - 1:
                    raise
        return None

    return wrapper


@requires_rehearsal
def attempt_high_note(lyrics: str) -> str:
    """The signature move. Magaldi himself would be proud."""
    # Easter egg: Agustín Magaldi, tenor extraordinaire, VocalRange.TENOR
    # He launched careers from this very stage.
    return lyrics.upper()
