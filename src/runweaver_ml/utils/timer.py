from contextlib import contextmanager
from time import perf_counter, time

@contextmanager
def timed_section(name: str, *, enabled: bool = False, callback=None):
    if not enabled:
        yield
        return

    start = perf_counter()

    try:
        yield
    finally:
        elapsed = perf_counter() - start

        if callback is not None:
            callback(name, elapsed)
        else:
            print(f"[Timing] {name}: {elapsed:.3f}s")



class Timer:
    def __init__(self, message="Working..."):
        self.message = message

    def __enter__(self):
        print(self.message, end="", flush=True)
        self.start = time()
        return self  # allows access if needed

    def __exit__(self, exc_type, exc, tb):
        elapsed = time() - self.start
        print(f" done ({elapsed:.2f}s)")
        self.elapsed = elapsed  # store if needed

class TimingAccumulator:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def __call__(self, name, elapsed):
        self.count += 1
        self.total += elapsed
        self.maximum = max(self.maximum, elapsed)

        if self.count % 100 == 0:
            print(
                f"[Timing] {name}: "
                f"avg={self.total / self.count:.3f}s "
                f"max={self.maximum:.3f}s "
                f"n={self.count}"
            )