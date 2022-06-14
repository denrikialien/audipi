# https://stackoverflow.com/a/33533514
# from __future__ import annotations

from dataclasses import dataclass
import soundfile as sf
import sounddevice as sd
from sounddevice import OutputStream
import numpy as np


@dataclass
class PlaybackInfo:
    data: np.ndarray
    sample_rate: float
    started_at: float


@dataclass
class PlaybackState:
    frame: int


@dataclass
class Playback:
    stream: OutputStream
    info: PlaybackInfo
    state: PlaybackState


def prepare(sound: np.ndarray, sample_rate: float) -> Playback:
    state = PlaybackState(0)
    info = PlaybackInfo(sound, sample_rate, 0.0)

    def callback(dst: np.ndarray, frames: int, *_):
        next_block(state, info, frames, dst)

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=sound.shape[1],
        callback=callback,
        finished_callback=finished_callback,
    )

    return Playback(stream, info, state)


def start(pb: Playback, start_frame: int):
    assert 0 <= start_frame < len(pb.info.data)
    assert not pb.stream.active
    assert not pb.stream.closed
    pb.state.frame = start_frame
    pb.stream.start()
    pb.info.started_at = pb.stream.time


def stop(pb: Playback):
    assert pb.stream.active
    assert not pb.stream.closed
    pb.stream.stop()
    pb.stream.close()


def current_time(pb: Playback) -> float:
    return pb.stream.time - pb.info.started_at


def next_block(
    state: PlaybackState,
    info: PlaybackInfo,
    block_size: int,
    dst: np.ndarray,
):
    sound = info.data
    block = min(block_size, len(sound) - state.frame)
    dst[:block] = sound[state.frame : state.frame + block]
    if block < block_size:
        dst[block:] = 0
        raise sd.CallbackStop()
    else:
        state.frame += block


# class _Playback:
#     def __init__(
#         self, sound: np.ndarray, samplerate: float, frame: int = 0
#     ) -> None:
#         assert 0 <= frame < len(sound)
#         stream = sd.OutputStream(
#             samplerate=samplerate,
#             channels=sound.shape[1],
#             callback=self.next_block,
#             finished_callback=self.on_end,
#         )
#         self.__audio = sound
#         self.__stream = stream
#         self.__current_frame = frame
#         self.__state = collections.deque(maxlen=1)
#         stream.start()
#         self.__started_at = stream.time

#     def on_end(self):
#         print(
#             "time: ", self.__stream.time - self.__started_at, " [s]"
#         )

#     def next_block(
#         self, output: np.ndarray, frames: int, time, status
#     ):
#         print(
#             "time: ", self.__stream.time - self.__started_at, " [s]"
#         )
#         current_frame = self.__current_frame
#         audio = self.__audio
#         chunk_size = min(frames, len(audio) - current_frame)
#         output[:chunk_size] = audio[
#             current_frame : current_frame + chunk_size
#         ]
#         if chunk_size < frames:
#             output[chunk_size:] = 0
#             raise sd.CallbackStop()
#         else:
#             self.__current_frame += chunk_size

#     def stop(self):
#         assert self.__stream.active
#         self.__stream.stop()
#         self.__stream.close()


def main(audiofile):
    audio, fs = sf.read(audiofile, always_2d=True)
    # _Playback(audio, samplerate=fs)
    pb = prepare(audio, fs)
    start(pb, 0)
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
