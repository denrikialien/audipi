from dataclasses import dataclass
from unit import msec, st_time
from audio import Audio
from sounddevice import OutputStream, CallbackStop
import soundfile
import numpy as np


@dataclass
class _State:
    buff_first_frame: int
    buff_last_frame: int
    buff_output_time: st_time


class Playback:
    _audio: Audio
    _loop: bool
    _start_position: msec
    _stream: OutputStream | None = None
    _state: _State | None = None

    def __init__(
        self,
        audio: Audio,
        start_position: msec,
        loop: bool,
    ):
        self._audio = audio
        self._start_position = start_position
        self._loop = loop

        assert 0 <= start_position <= self._audio.length

        self._stream = OutputStream(
            samplerate=audio.sample_rate,
            channels=audio.channels,
            callback=self.stream_callback,
        )

    def start(self):
        assert self._stream is not None
        assert not self._stream.active
        self._stream.start()

    def stream_callback(
        self, buff: np.ndarray, buff_size: int, time, _
    ):
        state = self._state
        first_frame = (
            self._audio.frame_at(self._start_position)
            if state is None
            else state.buff_last_frame + 1
        )
        last_frame, reached_end = _fill_buffer(
            buff,
            buff_size,
            first_frame,
            self._loop,
            self._audio,
        )
        if state is None:
            self._state = _State(
                buff_first_frame=first_frame,
                buff_last_frame=last_frame,
                buff_output_time=time.outputBufferDacTime,
            )
        else:
            state.buff_first_frame = first_frame
            state.buff_last_frame = last_frame
            state.buff_output_time = time.outputBufferDacTime
        if reached_end:
            raise CallbackStop()

    def abort(self):
        assert self._stream is not None
        self._stream.close()
        self._stream = None

    def stop(self):
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def loop_on(self):
        self._loop = True

    def loop_off(self):
        self._loop = False

    @property
    def loop(self) -> bool:
        return self._loop

    def current_position(self) -> msec | None:
        frame = self._estimate_current_frame()
        if frame is None:
            return None
        return self._audio.position_at(frame)

    def playing(self) -> bool:
        return self.current_position() is not None

    def _estimate_current_frame(self) -> int | None:
        state = self._state
        stream = self._stream
        if state is None or stream is None:
            return None
        dst = state.buff_output_time - stream.time
        return (
            state.buff_first_frame - dst * self._audio.sample_rate
            if dst > 0.0
            else None
        )


def _fill_buffer(
    buff: np.ndarray,
    buff_size: int,
    first_frame: int,
    loop: bool,
    audio: Audio,
) -> tuple[int, bool]:
    data = audio.data
    reached_end = False

    if loop:
        left, right = audio.section_at(first_frame)
        if right is None:
            right = audio.frames - 1
        if left is None:
            left = 0
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


if __name__ == "__main__":
    sample = "./local/sample.wav"
    # sample = "./local/loretta.wav"
    data, fs = soundfile.read(sample, always_2d=True)

    start_position = msec(0)
    audio = Audio(data, fs, _markers=[msec(5000), msec(10000)])
    loop = True
    pb = None

    while True:
        cmd = input("command > ")
        if cmd == "time" and pb is not None:
            time = pb.current_position()
            if time is not None:
                print("time: ", time / 1000, " [s]")
        elif cmd == "stop" and pb is not None:
            start_position = pb.current_position() or msec(0)
            pb.stop()
            pb = None
        elif cmd == "start" and pb is None:
            pb = Playback(audio, start_position, loop)
            pb.start()
        elif cmd == "quit":
            if pb is not None:
                pb.abort()
            break
        elif cmd == "loop on":
            loop = True
            if pb is not None:
                pb.loop_on()
        elif cmd == "loop off":
            loop = False
            if pb is not None:
                pb.loop_off()
