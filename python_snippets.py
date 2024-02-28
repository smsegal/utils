from functools import partial, wraps
import logging
import sys


def silence_output(_func=None, *, stdout=True, stderr=True, loggers=True):
    if _func is None:
        return partial(silence_output, stdout=stdout, stderr=stderr, loggers=loggers)

    @wraps(_func)
    def wrapper(*args, **kwargs):
        _stdout = sys.stdout
        _stderr = sys.stderr
        _loggers = logging.getLogger().handlers[:]

        sys.stdout = io.StringIO() if stdout else sys.stdout
        sys.stderr = io.StringIO() if stderr else sys.stderr
        if loggers:
            for handler in _loggers:
                logging.getLogger().removeHandler(handler)
            logging.getLogger().addHandler(logging.NullHandler())
        ret = _func(*args, **kwargs)
        sys.stdout = _stdout
        sys.stderr = _stderr
        logging.getLogger().handlers = _loggers
        return ret

    return wrapper


def set_repr():
    import torch
    global __SET_REPR  # noqa: PLW0603
    if "__SET_REPR" in globals() and __SET_REPR:  # type: ignore
        return

    # a better repr for debugging tensors, shows shape first.
    def custom_repr(self):
        return f"{{Shape:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr  # type: ignore
    __SET_REPR = True
