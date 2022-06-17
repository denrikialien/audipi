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
    _state: State
    _start_position: msec
    _st_time_offset: float
    _stream: OutputStream


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
    pl = Playing(
        sound=pb.sound,
        params=pb.params,
        _state=State(_pos2frame(pb.sound, pb.position)),
        _start_position=pb.position,
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
    start_time = current_position(pb)
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


def current_position(pb: Playing) -> msec:
    dt = (pb._stream.time - pb._st_time_offset) * 1000
    time = msec(int(dt) + pb._start_position)
    return min(time, length(pb.sound))


def seek(pb: Playing, position: msec):
    print("(pre) seek to ", position, " (len=", length(pb.sound))
    position = max(msec(0), min(position, length(pb.sound)))
    print("seek to ", position)
    pb._start_position = position
    pb._st_time_offset = pb._stream.time
    pb._state.last_buffered_frame = _pos2frame(pb.sound, position)


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


def _pos2frame(sound: Sound, position: msec) -> int:
    frame = int(sound.sample_rate * position / 1000)
    return max(0, min(frame, len(sound.data) - 1))


if __name__ == "__main__":
    sample = "./local/loretta.wav"
    audio, fs = soundfile.read(sample, always_2d=True)
    pb = prepare(audio, fs, start_time=msec(0), params=Params())
    pb = start(pb)
    while True:
        cmd = input("command > ")
        if cmd == "time" and isinstance(pb, Playing):
            print("time: ", current_position(pb) / 1000, " [s]")
        if cmd == "stop" and isinstance(pb, Playing):
            pb = stop(pb)
            print("stopped at: ", pb.position / 1000, " [s]")
        elif cmd == "start" and isinstance(pb, Pausing):
            pb = start(pb)
        elif cmd == "seek5" and isinstance(pb, Playing):
            seek(pb, msec(current_position(pb) + 5000))
        elif cmd == "seek-5" and isinstance(pb, Playing):
            seek(pb, msec(current_position(pb) - 5000))
        elif cmd == "quit":
            if isinstance(pb, Playing):
                stop(pb)
            break
