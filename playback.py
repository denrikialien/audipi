from dataclasses import dataclass
from typing import NewType
import typing
from sounddevice import OutputStream, CallbackStop
import soundfile
import numpy as np
from more_itertools import first_true

msec = NewType("msec", int)
st_time = NewType("st_time", float)


@dataclass
class Sound:
    data: np.ndarray
    sample_rate: float


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
class Playing:
    sound: Sound
    params: Params
    _stream: OutputStream
    _state: State | None = None


@dataclass
class Pausing:
    sound: Sound
    params: Params
    position: msec


def prepare(
    data: np.ndarray,
    sample_rate: float,
    start_time: msec,
    params: Params,
) -> Pausing:
    sound = Sound(data, sample_rate)
    return Pausing(
        sound=sound,
        params=params,
        position=start_time,
    )


def start(pb: Pausing) -> Playing:
    _pb = Playing(
        sound=pb.sound,
        params=pb.params,
        # init at the end of this function
        _stream=typing.cast(OutputStream, None),
    )

    def callback(buff: np.ndarray, buff_size: int, time, _):
        first_frame = (
            _pos2frame(_pb.sound, pb.position)
            if _pb._state is None
            else _pb._state.buff_last_frame + 1
        )
        last_frame, should_stop = _fill_buffer(
            _pb.sound,
            _pb.params,
            first_frame,
            buff_size,
            buff,
        )
        if _pb._state is None:
            _pb._state = State(
                buff_first_frame=first_frame,
                buff_last_frame=last_frame,
                buff_output_time=time.outputBufferDacTime,
            )
        else:
            _pb._state.buff_first_frame = first_frame
            _pb._state.buff_last_frame = last_frame
            _pb._state.buff_output_time = time.outputBufferDacTime
        if should_stop:
            raise CallbackStop()

    stream = OutputStream(
        samplerate=pb.sound.sample_rate,
        channels=pb.sound.data.shape[1],
        callback=callback,
    )
    stream.start()
    _pb._stream = stream
    return _pb


def stop(pb: Playing) -> Pausing:
    start_time = estimate_current_position(pb)
    assert start_time is not None
    stream = pb._stream
    if stream.active:
        stream.stop()
    stream.close()
    return prepare(
        pb.sound.data,
        pb.sound.sample_rate,
        start_time,
        pb.params,
    )


def length(sound: Sound) -> msec:
    length = len(sound.data) / sound.sample_rate * 1000
    return msec(int(length))


def estimate_current_position(pb: Playing) -> msec | None:
    frame = _estimate_current_frame(pb)
    if frame is None:
        return None
    return _frame2pos(pb.sound, frame)


def _estimate_current_frame(pb: Playing) -> int | None:
    state = pb._state
    if state is None:
        return None
    dst = state.buff_output_time - pb._stream.time
    return (
        state.buff_first_frame - dst * pb.sound.sample_rate
        if dst > 0.0
        else None
    )


def _fill_buffer(
    sound: Sound,
    params: Params,
    first_frame: int,
    buff_size: int,
    buff: np.ndarray,
) -> tuple[int, bool]:
    data = sound.data
    should_stop = False

    if params.loop:
        left, right = _region(first_frame, params.markers, sound)
        size = min(buff_size, right - first_frame + 1)
        last_frame = first_frame + size - 1
        buff[:size] = data[first_frame : last_frame + 1]
        if size < buff_size:
            # Expects that 'buff_size' is sufficiently small
            # than the region's size.
            last_frame = left + buff_size - size - 1
            buff[size:] = data[left : last_frame + 1]

    else:
        size = min(buff_size, len(data) - first_frame)
        last_frame = first_frame + size - 1
        buff[:size] = data[first_frame : last_frame + 1]
        if size < buff_size:
            buff[size:] = 0
            should_stop = True

    return last_frame, should_stop


def _pos2frame(sound: Sound, position: msec) -> int:
    frame = int(sound.sample_rate * position / 1000)
    return max(0, min(frame, len(sound.data) - 1))


def _frame2pos(sound: Sound, frame: int) -> msec:
    return msec(int(frame / sound.sample_rate * 1000))


def _region(
    frame: int, markers: list[msec], sound: Sound
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
    # sample = "./local/sample.wav"
    sample = "./local/loretta.wav"
    audio, fs = soundfile.read(sample, always_2d=True)
    pb = prepare(
        audio,
        fs,
        # start_time=msec(0),
        start_time=msec(6000),
        params=Params(loop=True, markers=[msec(5000), msec(10000)]),
        # params=Params(loop=False, markers=[]),
    )
    pb = start(pb)
    while True:
        cmd = input("command > ")
        if cmd == "time" and isinstance(pb, Playing):
            time = estimate_current_position(pb)
            if time is not None:
                print("time: ", time / 1000, " [s]")
        if cmd == "stop" and isinstance(pb, Playing):
            pb = stop(pb)
            print("stopped at: ", pb.position / 1000, " [s]")
        elif cmd == "start" and isinstance(pb, Pausing):
            pb = start(pb)
        elif cmd == "quit":
            if isinstance(pb, Playing):
                stop(pb)
            break
