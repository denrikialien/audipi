from dataclasses import dataclass
from typing import Any, TypeVar
import typing


class _Hole:
    def __init__(self):
        self.__dict__["_body"] = None
        self.__dict__["_filled"] = False

    def __setattr__(self, __name, __value):
        return self.__dict__["_body"].__setattr__(__name, __value)

    def __getattr__(self, __name):
        return self.__dict__["_body"].__getattribute__(__name)

    def __len__(self):
        return self.__dict__["_body"].__len__()


T = TypeVar("T")


def Hole() -> T:
    return typing.cast(T, _Hole())


def fill_hole(hole: Any, body: Any):
    if not isinstance(hole, _Hole):
        raise ValueError("Hole type is expected")
    if hole.__dict__["_filled"]:
        raise ValueError("The hole has been already filled")
    hole.__dict__["_body"] = body
    hole.__dict__["_filled"] = True


if __name__ == "__main__":

    @dataclass
    class Foo:
        x: int
        y: float

    foo: Foo = Hole()
    # print(foo.x, foo.y)  # <- AttributeError!
    fill_hole(foo, Foo(10, 3.14))
    print(foo.x, foo.y)
