import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import yaml


@dataclass
class Mob:
    id: int
    name: str


@dataclass
class CrateType:
    name: str
    aliases: List[str]


@dataclass
class ItemConfig:
    type: int
    type_name: str
    id: int
    name: str
    number_of_kills_till_item: Union[None, int]
    number_of_kills_till_item_per_mob: Union[None, Dict[Mob, int]]
    number_of_crates_till_item: Union[None, int]
    number_of_crates_till_item_per_crate_type: Union[None, Dict[CrateType, int]]


@dataclass
class Config:
    base: Path
    patches: List[Path]
    xdt: Path
    items: List[ItemConfig]


class KnowledgeBase:
    def __init__(self) -> None:
        pass


def alter_chances():
    pass


def get_operating_constraints():
    pass


def standardize_config():
    pass


def save_new_drops():
    pass


def construct_knowledge_base(config_path: Path) -> KnowledgeBase:
    with open(config_path) as r:
        config_data = yaml.load(r, yaml.SafeLoader)


def main():
    config_path = Path('rarifyconfig.yml' if len(sys.argv) < 2 else sys.argv[1])
    knowledge_base = construct_knowledge_base(config_path)


if __name__ == '__main__':
    main()
