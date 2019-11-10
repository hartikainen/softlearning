"""Provides functions that are utilized by the command line interface.

In particular, the examples are exposed to the command line interface
(defined in `softlearning.scripts.console_scripts`) through the
`get_trainable_class`, `get_variant_spec`, and `get_parser` functions.
"""


def get_trainable_class(*args, **kwargs):
    from .main import DiaynExperimentRunner
    return DiaynExperimentRunner


def get_variant_spec(command_line_args, *args, **kwargs):
    from .variants import get_variant_spec
    variant_spec = get_variant_spec(command_line_args, *args, **kwargs)
    return variant_spec


def get_parser():
    from examples.utils import get_parser
    parser = get_parser()
    return parser
