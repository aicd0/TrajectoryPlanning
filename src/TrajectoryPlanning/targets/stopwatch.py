import time
from utils.stopwatch import Stopwatch
from utils.test import AST_EQUAL_FLT

def main():
    stopwatch = Stopwatch()
    AST_EQUAL_FLT(stopwatch.span(), 0)
    stopwatch.start()
    AST_EQUAL_FLT(stopwatch.span(), 0)
    time.sleep(1)
    AST_EQUAL_FLT(stopwatch.span(), 1, error=1e-2)
    stopwatch.pause()
    AST_EQUAL_FLT(stopwatch.span(), 1, error=1e-2)
    time.sleep(1)
    AST_EQUAL_FLT(stopwatch.span(), 1, error=1e-2)
    stopwatch.start()
    AST_EQUAL_FLT(stopwatch.span(), 1, error=1e-2)
    time.sleep(1)
    AST_EQUAL_FLT(stopwatch.span(), 2, error=1e-2)
    stopwatch.reset()
    AST_EQUAL_FLT(stopwatch.span(), 0)
    time.sleep(1)
    AST_EQUAL_FLT(stopwatch.span(), 0)