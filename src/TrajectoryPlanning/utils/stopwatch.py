import time

class Stopwatch:
    def __init__(self) -> None:
        self.__start = False
        self.__span = 0
        self.__begin = None

    def start(self) -> None:
        if not self.__start:
            self.__start = True
            self.__begin = time.time()

    def pause(self) -> None:
        t = time.time()
        if self.__start:
            self.__span += t - self.__begin
            self.__start = False

    def reset(self) -> None:
        self.__start = False
        self.__span = 0

    def span(self) -> float:
        ans = self.__span
        if self.__start:
            ans += time.time() - self.__begin
        return ans