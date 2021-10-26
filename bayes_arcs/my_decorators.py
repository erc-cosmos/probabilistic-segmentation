"""Some custom decorators."""

import functools
import inspect


def preprocess(kw, preprocessor):
    """Decorate to preprocess a given argument."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = inspect.getcallargs(func, *args, **kwargs)
            kwargs[kw] = preprocessor(kwargs[kw])
            return func(**kwargs)
        return wrapper
    return decorator


def single_or_list(func=None, *, kw=None):
    """
    Decorate to convert a single value to a list of length 1.

    If kw is specified, conversion is applied to this argument, otherwise to the first argument
    """
    if func is None:
        return functools.partial(single_or_list, kw=kw)

    def ensure_list(maybe_list_arg):
        return [maybe_list_arg] if not isinstance(maybe_list_arg, list) else maybe_list_arg
    if kw is None:
        @functools.wraps(func)
        def wrapper(maybe_list_arg, *args, **kwargs):
            return func(ensure_list(maybe_list_arg), *args, **kwargs)
        return wrapper
    else:
        return preprocess(kw, ensure_list)(func)
