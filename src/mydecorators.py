"""Some custom decorators"""

import functools
import inspect

def preprocess(kw,preprocessor):
    """ Decorator to preprocess a given argument """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            kwargs = inspect.getcallargs(func,*args,**kwargs)
            kwargs[kw] = preprocessor(kwargs[kw])
            return func(**kwargs)
        return wrapper
    return decorator


def singleOrList(func=None, *, kw=None):
    """
    Decorator to convert a single value to a list of length 1 
    If kw is specified, conversion is applied to this argument, otherwise to the first argument
    """
    if func is None:
        return functools.partial(singleOrList,kw=kw)

    def ensureList(maybeListArg):
        return [maybeListArg] if not isinstance(maybeListArg, list) else maybeListArg
    if kw is None:
        @functools.wraps(func)
        def wrapper(maybeListArg,*args,**kwargs):
            return func(ensureList(maybeListArg),*args,**kwargs)
        return wrapper
    else:
        return preprocess(kw,ensureList)(func)