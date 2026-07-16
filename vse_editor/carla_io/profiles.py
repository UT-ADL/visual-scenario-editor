"""Connection profiles: how VSE talks to a CARLA server (moved from vse.py, step-34)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConnectionProfile:
    """Describe how VSE should talk to a CARLA server."""

    name: str
    host: str
    port: int
    manage_server: bool
    display_name: str
    description: str = ""
    map_hint: Optional[str] = None

    @property
    def is_remote(self) -> bool:
        return not self.manage_server
