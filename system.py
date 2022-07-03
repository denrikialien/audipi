from dataclasses import dataclass
from playback import Playback


@dataclass
class SystemState:
    pb: Playback


def foo(x: int):
    def rag(y: int):
        return x * y

    return rag


times2 = foo(2)
z = times2(5)
