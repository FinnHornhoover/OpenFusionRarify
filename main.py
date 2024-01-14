import sys
import logging
from pathlib import Path

from rarify.file_ops import load_data, save_new_drops
from rarify.solver import alter_chances


def main() -> None:
    """
    Entry point of the program.

    This program takes a configuration defined by a `.yml` or `.yaml` file of the
    expected format, figures out where to apply which change, and alters the drops
    in the specified way.

    Run directly with `main.py` or `rarify.exe` if the config file is named
    `rarifyconfig.yml` and is in the same directory as `main.py` or `rarify.exe`.

    You can also give the path to the YAML file as an argument, or just drag the YAML
    file onto the executable to run it with that YAML file.
    """
    logging.basicConfig(
        filename='rarify.log',
        filemode='w',
        level=logging.INFO,
        format='[%(levelname)7s @ %(asctime)s] %(message)s'
    )

    config_path = Path('rarifyconfig.yml' if len(sys.argv) < 2 else sys.argv[1])

    knowledge_base, config, item_configs = load_data(config_path)

    alter_chances(knowledge_base, item_configs)
    save_new_drops(knowledge_base, config)


if __name__ == '__main__':
    main()
