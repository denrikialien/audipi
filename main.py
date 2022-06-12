# https://stackoverflow.com/a/33533514
# from __future__ import annotations

import collections
import soundfile as sf
import sounddevice as sd
import numpy as np


class SoundBox:
    def __init__(
        self, audio: np.ndarray, samplerate: float, frame: int = 0
    ) -> None:
        assert 0 <= frame < len(audio)
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=audio.shape[1],
            callback=self.on_process,
        )
        self.__audio = audio
        self.__stream = stream
        self.__current_frame = frame
        self.__state = collections.deque(maxlen=1)
        stream.start()

    def on_process(
        self, output: np.ndarray, frames: int, time, status
    ):
        current_frame = self.__current_frame
        audio = self.__audio
        chunk_size = min(frames, len(audio) - current_frame)
        output[:chunk_size] = audio[
            current_frame : current_frame + chunk_size
        ]
        if chunk_size < frames:
            output[chunk_size:] = 0
            raise sd.CallbackStop()
        else:
            self.__current_frame += chunk_size

    def stop(self):
        assert self.__stream.active
        self.__stream.stop()
        self.__stream.close()


def main(audiofile):
    audio, fs = sf.read(audiofile, always_2d=True)
    SoundBox(audio, samplerate=fs)
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
