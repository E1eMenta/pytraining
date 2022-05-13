import argparse
import os
from importlib import import_module
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to config file')
    args = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise Exception(f"Config file {args.config} is not exists")

    config_file = Path(args.config[:-3])
    config_file = config_file.as_posix().replace('/', '.')

    a = import_module(config_file)
    config = a.Config()
    config.run()
