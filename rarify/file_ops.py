import json
from pathlib import Path
from typing import List, Tuple, Optional

import yaml

from .config import Config, ItemConfig
from .drops import generate_patch
from .knowledge_base import KnowledgeBase


def load_data(
    config_path: Path,
) -> Tuple[KnowledgeBase, Config, Optional[List[ItemConfig]]]:
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
        config_data = yaml.safe_load(r)

    config = Config(
        base=Path(config_data["base"]),
        patches=[Path(patch_path) for patch_path in (config_data.get("patches") or [])],
        xdt=Path(config_data["xdt"]),
        output=Path(config_data["output"]),
        generate_patch=bool(config_data["generate-patch"]),
    )

    knowledge_base = KnowledgeBase(config)
    item_configs = [
        ItemConfig.from_dict(knowledge_base, data) for data in config_data["items"]
    ]

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
