from dataclasses import dataclass
from unit import msec
import numpy as np
from more_itertools import first_true


@dataclass
class Audio:
    _data: np.ndarray
    _sample_rate: float
    _markers: list[msec]

    @property
    def length(self) -> msec:
        length = self.frames / self._sample_rate * 1000
        return msec(int(length))

    @property
    def channels(self) -> int:
        return self._data.shape[1]

    @property
    def frames(self) -> int:
        return len(self._data)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def frame_at(self, position: msec) -> int:
        assert 0 <= position <= self.length
        return int(
            np.clip(
                self._sample_rate * position / 1000,
                0,
                len(self._data) - 1,
            )
        )

    def position_at(self, frame: int) -> msec:
        assert 0 <= frame < self.frames
        return msec(int(frame / self._sample_rate * 1000))

    def set_marker(self, position: msec):
        ix = first_true(
            range(len(self._markers)),
            pred=lambda ix: self._markers[ix] >= position,
            default=None,
        )
        if ix is None:
            self._markers.append(position)
        elif self._markers[ix] != position:
            self._markers.insert(ix, position)

    def unset_marker(self, position: msec):
        self._markers.remove(position)

    def section_at(self, frame: int) -> tuple[int | None, int | None]:
        markers = self._markers
        if len(markers) == 0:
            return (0, self.frames - 1)

        ix = first_true(
            (ix for ix in range(len(markers))),
            pred=lambda ix: self.frame_at(markers[ix]) > frame,
            default=None,
        )

        if ix is None:
            return (self.frame_at(markers[-1]), None)
        if ix == 0:
            return (None, self.frame_at(markers[0]))
        else:
            return (
                self.frame_at(markers[ix - 1]),
                self.frame_at(markers[ix]),
            )
