from .base import System
from .pendulum import Pendulum

SYSTEMS = {"pendulum": Pendulum}


def get_system(name: str) -> System:
    try:
        return SYSTEMS[name]()
    except KeyError:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}") from None
