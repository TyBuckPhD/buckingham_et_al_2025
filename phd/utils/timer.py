import time
from functools import wraps, partial

class Timer:
    def __init__(self, func=None, name=None):
        self.func = func
        self.name = name
        if func:
            wraps(func)(self)

    def __call__(self, *args, **kwargs):
        if self.func:
            return self._call(*args, **kwargs)
        else:
            # When using Timer without () as a decorator
            return partial(self._call, *args, **kwargs)

    def _call(self, *args, **kwargs):
        if self.func is None:
            self.func, args = args[0], args[1:]
            wraps(self.func)(self)

        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        minutes, seconds = divmod(execution_time, 60)
        func_name = self.name if self.name else self.func.__name__
        print(f"Passed: {func_name} took {int(minutes):02d}:{int(seconds):02d} mm:ss to run.")

        return result

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)
    