import configparser
import json
import sys
import importlib


def config_to_kwargs(config: configparser.ConfigParser, section: str, definitions: dict):
    kwargs = {}
    for key, definition in definitions.items():
        if isinstance(definition, tuple):
            dtype, default = definition
        else:
            dtype, default = definition, object()  # default for config is configparser._UNSET  (which is == object())
        if dtype is int:
            value = config.getint(section, key, fallback=default)
        elif dtype is float:
            value = config.getfloat(section, key, fallback=default)
        elif dtype is bool:
            value = config.getboolean(section, key, fallback=default)
        else:
            value = config.get(section, key, fallback=default)
            try:
                value = dtype(value)
            except TypeError:
                pass
        kwargs[key] = value
    return kwargs


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def instantiate(path, *args, **kwargs):
    # If path is a path to json config file
    # Then instantiate from json
    if '.json' in path:
        with open(path) as f:
            config = json.loads(f.read())  # type: dict
            class_name = config.pop('class')  # type: str

            # Update the kwargs from config (rewriting function kwargs)
            kwargs.update(config)
    else:
        # Otherwise instantiate assuming that path is a path to the class
        class_name = path

    parts = class_name.split('.')
    if len(parts) == 1:
        cls_obj = class_for_name(None, parts[-1])
    else:
        cls_obj = class_for_name('.'.join(parts[:-1]), parts[-1])

    return cls_obj(*args, **kwargs)
