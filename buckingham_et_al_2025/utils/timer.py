import time
from functools import wraps, partial


class Timer:
    """
    Timer decorator for measuring function execution time.

    This decorator can be applied to any function to measure the time it takes to execute.
    When the decorated function is called, the Timer records the start and end times, computes
    the elapsed time, and prints the result in minutes and seconds in a formatted message.

    The Timer supports usage both with and without arguments. When used without parentheses,
    it directly wraps the function. When used with parentheses, it can accept an optional 'name'
    parameter to override the function's name in the printed output.

    Attributes:
        func (callable): The function being wrapped.
        name (str): Optional name to display; defaults to the wrapped function's __name__ if not provided.

    Methods:
        __call__(*args, **kwargs):
            Calls the wrapped function, measures its execution time, and prints the elapsed time.
        _call(*args, **kwargs):
            Internal method that performs timing, function execution, and prints the execution time.
        __get__(instance, owner):
            Supports instance methods by returning a partial function bound to the instance.

    Usage Examples:
        @Timer
        def my_function(...):
            ...

        # Or with a custom name:
        @Timer(name="CustomFunction")
        def my_function(...):
            ...
    """

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
        print(
            f"Passed: {func_name} took {int(minutes):02d}:{int(seconds):02d} mm:ss to run."
        )

        return result

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)
