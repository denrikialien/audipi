from dataclasses import dataclass
from typing import NewType
from sounddevice import OutputStream, CallbackStop

import soundfile
import numpy as np
from more_itertools import first_true

msec = NewType("msec", int)
st_time = NewType("st_time", float)


@dataclass
class Audio:
    data: np.ndarray
    sample_rate: float

    @property
    def length(self) -> msec:
        length = len(self.data) / self.sample_rate * 1000
        return msec(int(length))


@dataclass
class Params:
    markers: list[msec]
    loop: bool


@dataclass
class State:
    buff_first_frame: int
    buff_last_frame: int
    buff_output_time: st_time


@dataclass
class Playback:
    _audio: Audio
    _params: Params
    _stream: OutputStream | None = None
    _state: State | None = None


# @dataclass
# class Pausing:
#     sound: Sound
#     params: Params
#     position: msec


# def prepare(
#     data: np.ndarray,
#     sample_rate: float,
#     start_time: msec,
#     params: Params,
# ) -> Pausing:
#     sound = Sound(data, sample_rate)
#     return Pausing(
#         sound=sound,
#         params=params,
#         position=start_time,
#     )


def open(
    data: np.ndarray,
    sample_rate: float,
    start_position: msec,
    params: Params,
) -> Playback:
    pb = Playback(_audio=Audio(data, sample_rate), _params=params)
    assert 0 <= start_position <= pb._audio.length

    def callback(buff: np.ndarray, buff_size: int, time, _):
        first_frame = (
            _pos2frame(pb._audio, start_position)
            if pb._state is None
            else pb._state.buff_last_frame + 1
        )
        last_frame, reached_end = _fill_buffer(
            pb._audio,
            pb._params,
            first_frame,
            buff_size,
            buff,
        )
        if pb._state is None:
            pb._state = State(
                buff_first_frame=first_frame,
                buff_last_frame=last_frame,
                buff_output_time=time.outputBufferDacTime,
            )
        else:
            pb._state.buff_first_frame = first_frame
            pb._state.buff_last_frame = last_frame
            pb._state.buff_output_time = time.outputBufferDacTime
        if reached_end:
            raise CallbackStop()

    stream = OutputStream(
        samplerate=pb._audio.sample_rate,
        channels=pb._audio.data.shape[1],
        callback=callback,
    )
    stream.start()
    pb._stream = stream
    return pb


def close(pb: Playback):
    assert pb._stream is not None
    pb._stream.close()
    pb._stream = None


def set_marker(pb: Playback, position: msec):
    markers = pb._params.markers
    ix = first_true(
        range(len(markers)),
        pred=lambda ix: markers[ix] >= position,
        default=None,
    )
    if ix is None:
        markers.append(position)
    elif markers[ix] != position:
        markers.insert(ix, position)


def unset_marker(pb: Playback, position: msec):
    pb._params.markers.remove(position)


def loop_on(pb: Playback):
    pb._params.loop = True


def loop_off(pb: Playback):
    pb._params.loop = False


def _current_position(pb: Playback) -> msec | None:
    frame = _estimate_current_frame(pb)
    if frame is None:
        return None
    return _frame2pos(pb._audio, frame)


def playing(pb: Playback) -> bool:
    return _current_position(pb) is not None


def _estimate_current_frame(pb: Playback) -> int | None:
    state = pb._state
    stream = pb._stream
    if state is None or stream is None:
        return None
    dst = state.buff_output_time - stream.time
    return (
        state.buff_first_frame - dst * pb._audio.sample_rate
        if dst > 0.0
        else None
    )


def _fill_buffer(
    sound: Audio,
    params: Params,
    first_frame: int,
    buff_size: int,
    buff: np.ndarray,
) -> tuple[int, bool]:
    data = sound.data
    reached_end = False

    if params.loop:
        left, right = _section(first_frame, params.markers, sound)
        size = min(buff_size, right - first_frame + 1)
        last_frame = first_frame + size - 1
        buff[:size] = data[first_frame : last_frame + 1]
        if size < buff_size:
            # Expects that 'buff_size' is sufficiently small
            # than the size of the current section.
            last_frame = left + buff_size - size - 1
            buff[size:] = data[left : last_frame + 1]

    else:
        size = min(buff_size, len(data) - first_frame)
        last_frame = first_frame + size - 1
        buff[:size] = data[first_frame : last_frame + 1]
        if size < buff_size:
            buff[size:] = 0
            reached_end = True

    return last_frame, reached_end


def _pos2frame(sound: Audio, position: msec) -> int:
    frame = int(sound.sample_rate * position / 1000)
    return np.clip(frame, 0, len(sound.data) - 1)


def _frame2pos(sound: Audio, frame: int) -> msec:
    return msec(int(frame / sound.sample_rate * 1000))


def _section(
    frame: int, markers: list[msec], sound: Audio
) -> tuple[int, int]:
    if len(markers) == 0:
        return (0, len(sound.data) - 1)

    right = first_true(
        (ix for ix in range(len(markers))),
        pred=lambda ix: _pos2frame(sound, markers[ix]) > frame,
        default=None,
    )

    if right is None:
        return (_pos2frame(sound, markers[-1]), len(sound.data) - 1)
    if right == 0:
        return (0, _pos2frame(sound, markers[0]))
    else:
        return (
            _pos2frame(sound, markers[right - 1]),
            _pos2frame(sound, markers[right]),
        )


if __name__ == "__main__":
    sample = "./local/sample.wav"
    # sample = "./local/loretta.wav"
    audio, fs = soundfile.read(sample, always_2d=True)

    start_position = msec(0)
    params = Params(loop=True, markers=[msec(5000), msec(10000)])
    pb = open(audio, fs, start_position, params)

    while True:
        cmd = input("command > ")
        if cmd == "time" and pb is not None:
            time = _current_position(pb)
            if time is not None:
                print("time: ", time / 1000, " [s]")
        if cmd == "stop" and pb is not None:
            start_position = _current_position(pb) or msec(0)
            close(pb)
            pb = None
        elif cmd == "start" and pb is None:
            pb = open(audio, fs, start_position, params)
        elif cmd == "quit":
            if pb is not None:
                close(pb)
            break
