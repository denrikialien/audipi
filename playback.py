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
    buff_output_time: float


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
        _state=State(
            buff_last_frame=-1,
            buff_first_frame=-1,
            buff_output_time=-1.0,
        ),
        _start_position=pb.position,
        # init at the end of this function
        _st_time_offset=typing.cast(float, None),
        _stream=typing.cast(OutputStream, None),
    )

    def callback(dst: np.ndarray, block_size: int, time, _):
        _next_block(pl.sound, pl.params, pl._state, block_size, dst)
        pl._state.buff_output_time = time.outputBufferDacTime

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
    start_time = estimate_current_position(pb)
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


def estimate_current_position(pb: Playing) -> msec:
    return _frame2pos(pb.sound, _estimate_current_frame(pb))


def _estimate_current_frame(pb: Playing) -> int:
    dst = pb._state.buff_output_time - pb._stream.time
    return pb._state.buff_first_frame - dst * pb.sound.sample_rate


def seek(pb: Playing, position: msec):
    position = max(msec(0), min(position, length(pb.sound)))
    pb._start_position = position
    pb._st_time_offset = pb._stream.time
    pb._state.buff_last_frame = _pos2frame(pb.sound, position)


def _next_block(
    sound: Sound,
    params: Params,
    state: State,
    block_size: int,
    dst: np.ndarray,
):
    first_frame = state.buff_last_frame + 1
    data = sound.data

    # if params.loop:
    #     left, right = _region(frame, params.markers, sound)
    #     region = data[left : right + 1]
    #     _next_block_looped(
    #         region,
    #         frame=frame - left,
    #         size=block_size,
    #         dst=dst,
    #     )
    #     state.next_buffered_frame = left + (frame + block_size) % len(
    #         region
    #     )

    if params.loop:
        left, right = _region(first_frame, params.markers, sound)
        size = min(block_size, right - first_frame + 1)
        last_frame = first_frame + size - 1
        dst[:size] = data[first_frame : last_frame + 1]
        if size < block_size:
            # Expects that 'block_size' is sufficiently small
            # than the region's size.
            last_frame = left + block_size - size - 1
            dst[size:] = data[left : last_frame + 1]
        state.buff_first_frame = first_frame
        state.buff_last_frame = last_frame

    else:
        size = min(block_size, len(data) - first_frame)
        last_frame = first_frame + size - 1
        dst[:size] = data[first_frame : last_frame + 1]
        state.buff_first_frame = first_frame
        state.buff_last_frame = last_frame
        if size < block_size:
            dst[size:] = 0
            raise CallbackStop()


def _next_block_looped(
    region: np.ndarray,
    frame: int,
    size: int,
    dst: np.ndarray,
):
    chunk = min(size, len(region) - frame)
    dst[:chunk] = region[frame : frame + chunk]
    if size > chunk:
        _next_block_looped(
            region,
            frame=0,
            size=size - chunk,
            dst=dst[chunk:],
        )


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
        start_time=msec(0),
        params=Params(loop=True, markers=[msec(5000)]),
    )
    pb = start(pb)
    while True:
        cmd = input("command > ")
        if cmd == "time" and isinstance(pb, Playing):
            print(
                "time: ", estimate_current_position(pb) / 1000, " [s]"
            )
        if cmd == "stop" and isinstance(pb, Playing):
            pb = stop(pb)
            print("stopped at: ", pb.position / 1000, " [s]")
        elif cmd == "start" and isinstance(pb, Pausing):
            pb = start(pb)
        elif cmd == "seek5" and isinstance(pb, Playing):
            seek(pb, msec(estimate_current_position(pb) + 5000))
        elif cmd == "seek-5" and isinstance(pb, Playing):
            seek(pb, msec(estimate_current_position(pb) - 5000))
        elif cmd == "quit":
            if isinstance(pb, Playing):
                stop(pb)
            break
