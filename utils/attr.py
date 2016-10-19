import functools
from abc import abstractproperty


def override(function):
    """Marker attribute to tell we are overriding a method
    """
    return function


def lazy_property(function):
    """A lazy property that is only evaluated once.
       Reduces clutter code and is similar to a singleton implementation for a property.
       
       References: http://danijar.com/structuring-your-tensorflow-models/
    """
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def lazy_abstractproperty(function):
    """A lazy abstractproperty that is only evaluated once.
       Reduces clutter code and is similar to a singleton implementation for a property.
       
       References: http://danijar.com/structuring-your-tensorflow-models/
    """
    attribute = '_' + function.__name__

    @abstractproperty
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper