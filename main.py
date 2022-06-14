# https://stackoverflow.com/a/33533514
# from __future__ import annotations

import soundfile as sf
from sounddevice import OutputStream, CallbackStop
import numpy as np


class Playback:
    __data: np.ndarray
    __sample_rate: float
    __frame: int
    __started_at: float
    __stream: OutputStream

    def __init__(
        self, data: np.ndarray, sample_rate: float, start_frame: int
    ):
        assert 0 <= start_frame < len(data)
        self.__data = data
        self.__sample_rate = sample_rate
        self.__frame = start_frame
        self.__stream = OutputStream(
            samplerate=self.__sample_rate,
            channels=self.__data.shape[1],
            callback=self.__next_block,
            finished_callback=self.stop,
        )
        self.__stream.start()
        self.__started_at = self.__stream.time

    def __next_block(self, dst: np.ndarray, block_size: int, *_):
        frame = self.__frame
        data = self.__data
        size = min(block_size, len(data) - frame)
        dst[:size] = data[frame : frame + size]
        if size < block_size:
            dst[size:] = 0
            raise CallbackStop()
        else:
            self.__frame += size

    def stop(self):
        stream = self.__stream
        if stream.active:
            stream.stop()
        stream.close()

    @property
    def time(self) -> float:
        return self.__stream.time - self.__started_at


def main(audiofile):
    audio, fs = sf.read(audiofile, always_2d=True)
    Playback(audio, sample_rate=fs, start_frame=0)
    while True:
        _ = input("press any key...")
        break


# def prepare_playback(audiofile):
#     audio, fs = sf.read(audiofile, always_2d=True)


# def start_playback():
#     while True:
#         cmd = get_command()
#         status = interpret_command(cmd)
#         if status is None:
#             break


# def interpret_command(cmd):
#     match cmd:
#         case "pause":
#             return None
#         case "resume":
#             return None
#         case _:
#             return None


# def get_command() -> str | None:
#     try:
#         return input("Command? > ")
#     except EOFError:
#         return None


if __name__ == "__main__":
    main(input("file? > "))
