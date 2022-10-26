import yaml
import logging


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_logger(name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    return logger
