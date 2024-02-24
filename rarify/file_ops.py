import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from rarify.config import Config, ItemConfig
from rarify.drops import generate_patch
from rarify.knowledge_base import KnowledgeBase


def load_data(
    config_path: Path,
) -> Tuple[KnowledgeBase, Config, List[ItemConfig]]:
    """
    Given a path, loads and constructs a knowledge base, program configuration and item
    configurations.

    Parameters
    ----------
    `config_path`: `Path`
        A fully resolved path pointing to the main YAML file.

    Returns
    -------
    `Tuple[KnowledgeBase, Config, List[ItemConfig]]`
        The knowledge base, program configuration and item configurations.
    """
    with open(config_path) as r:
        config_data: Dict[str, Any] = yaml.safe_load(r)

    config = Config.from_dict(config_data)
    knowledge_base = KnowledgeBase(config)
    item_configs = [
        ItemConfig.from_dict(knowledge_base, data)
        for data in config_data.get("items", [])
    ]
    item_configs = [item_config for item_config in item_configs if item_config]

    return knowledge_base, config, item_configs


def save_new_drops(knowledge_base: KnowledgeBase, config: Config) -> None:
    """
    Saves the altered drops object in the knowledge base, according to the config
    object.

    Parameters
    ----------
    `knowledge_base`: `KnowledgeBase`
        The knowledge base object which contains unaltered and altered drops objects.
    `config`: `Config`
        The config object which contains the saving directory and the option to
        generate a patch or not.
    """
    object_to_save = (
        generate_patch(knowledge_base.base_drops, knowledge_base.drops)
        if config.generate_patch
        else knowledge_base.drops
    )
    path_to_save = (
        config.output / "new_patch" if config.generate_patch else config.output
    )

    path_to_save.mkdir(parents=True, exist_ok=True)
    with open(path_to_save / "drops.json", "w") as w:
        json.dump(object_to_save, w, indent=4)
