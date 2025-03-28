# -----------------------------------------------------------------------------
# Functions for parsing args
# -----------------------------------------------------------------------------
import yaml
import os


class CfgNode(dict):
    """
    A dictionary subclass that allows access to its keys as attributes.
    """
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def load_cfg_from_cfg_file(file):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    return CfgNode(cfg_from_file)

