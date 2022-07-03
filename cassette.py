from typing import NewType
from dataclasses import dataclass
from more_itertools import first_true
import numpy as np


msec = NewType("msec", int)
ONE_SEC = msec(1000)


class MarkSet:
    _marks: list[msec]
    _audio_length: msec

    def __init__(self, audio_length: msec) -> None:
        self._marks = []
        self._audio_length = audio_length

    def set_at(self, position: msec):
        assert 0 < position < self._audio_length
        ix = first_true(
            range(len(self._marks)),
            pred=lambda ix: self._marks[ix] >= position,
            default=None,
        )
        if ix is None:
            self._marks.append(position)
        elif self._marks[ix] != position:
            self._marks.insert(ix, position)

    def unset_at(self, position: msec):
        self._marks.remove(position)

    def __getitem__(self, index: int) -> msec:
        return self._marks[index]

    def __len__(self) -> int:
        return len(self._marks)

    @property
    def audio_length(self) -> msec:
        return self._audio_length


@dataclass
class Audio:
    _data: np.ndarray
    _sample_rate: float

    @property
    def length(self) -> msec:
        length = self.frames / self._sample_rate * ONE_SEC
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
                self._sample_rate * position / ONE_SEC,
                0,
                len(self._data) - 1,
            )
        )

    def position_at(self, frame: int) -> msec:
        assert 0 <= frame < self.frames
        return msec(int(frame / self._sample_rate * ONE_SEC))


# def slice_section(
#     contains: msec,
#     audio: Audio,
#     marks: MarkSet,
# ) -> np.ndarray:
#     if len(marks) == 0:
#         return audio.data

#     ix = first_true(
#         (ix for ix in range(len(marks))),
#         pred=lambda ix: marks[ix] > contains,
#         default=None,
#     )

#     if ix is None:
#         return audio.data[audio.frame_at(marks[-1]) :]
#     if ix == 0:
#         return audio.data[: audio.frame_at(marks[0])]
#     else:
#         start = audio.frame_at(marks[ix - 1])
#         end = audio.frame_at(marks[ix])
#         return audio.data[start:end]


def slice_sections(audio: Audio, marks: MarkSet):
    assert audio.length == marks.audio_length
    start_pos = msec(0)
    for end_pos in marks:
        start = audio.frame_at(start_pos)
        end = audio.frame_at(end_pos)
        yield audio.data[start:end]
        start_pos = end_pos
    yield audio.data[start_pos:]
