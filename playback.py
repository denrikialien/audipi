from dataclasses import dataclass
from typing import NewType
import typing
from sounddevice import OutputStream, CallbackStop
import soundfile
import numpy as np

msec = NewType("msec", int)


@dataclass
class Sound:
    data: np.ndarray
    sample_rate: float


@dataclass
class Params:
    pass


@dataclass
class State:
    last_buffered_frame: int


@dataclass
class Playing:
    sound: Sound
    params: Params
    start_time: msec
    _state: State
    _st_time_offset: float
    _stream: OutputStream


@dataclass
class Pausing:
    sound: Sound
    params: Params
    time: msec


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
        time=start_time,
    )


def start(pb: Pausing) -> Playing:
    pl = Playing(
        sound=pb.sound,
        params=pb.params,
        start_time=pb.time,
        _state=State(_start_frame(pb.sound, pb.time)),
        _st_time_offset=typing.cast(float, None),  # init later
        _stream=typing.cast(OutputStream, None),
    )

    def callback(dst: np.ndarray, block_size: int, *_):
        _next_block(pl.sound, pl.params, pl._state, block_size, dst)

    stream = OutputStream(
        samplerate=pb.sound.sample_rate,
        channels=pb.sound.data.shape[1],
        callback=callback,
    )
    stream.start()
    pl._st_time_offset = stream.time
    pl._stream = stream
    return pl


def stop(pb: Playing) -> Pausing:
    start_time = current_time(pb)
    stream = pb._stream
    # if stream.active:
    #     stream.stop()
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


def current_time(pb: Playing) -> msec:
    dt = (pb._stream.time - pb._st_time_offset) * 1000
    time = msec(int(dt) + pb.start_time)
    return min(time, length(pb.sound))


def _next_block(
    sound: Sound,
    params: Params,
    state: State,
    block_size: int,
    dst: np.ndarray,
):
    frame = state.last_buffered_frame
    data = sound.data
    size = min(block_size, len(data) - frame)
    dst[:size] = data[frame : frame + size]
    if size < block_size:
        dst[size:] = 0
        raise CallbackStop()
    else:
        state.last_buffered_frame += size


def _start_frame(sound: Sound, start_time: msec) -> int:
    frame = int(sound.sample_rate * start_time / 1000)
    return max(0, min(frame, len(sound.data) - 1))


if __name__ == "__main__":
    sample = "./local/sample.wav"
    audio, fs = soundfile.read(sample, always_2d=True)
    pb = prepare(audio, fs, start_time=msec(0), params=Params())
    pb = start(pb)
    while True:
        cmd = input("command > ")
        if cmd == "time" and isinstance(pb, Playing):
            print("time: ", current_time(pb) / 1000, " [s]")
        if cmd == "stop" and isinstance(pb, Playing):
            pb = stop(pb)
            print("stopped at: ", pb.time / 1000, " [s]")
        elif cmd == "start" and isinstance(pb, Pausing):
            pb = start(pb)
        elif cmd == "quit":
            if isinstance(pb, Playing):
                stop(pb)
            break
