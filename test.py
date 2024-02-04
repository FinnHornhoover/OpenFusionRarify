import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
from argparse import ArgumentParser

import yaml

from rarify.drops import Drops
from rarify.config import OTHER_KEYWORDS

ZR_SEED = 123123
ITEM_SEED = 321321


def generate_zero_rarity_patch(
    drops: Drops, output_path: Path, patch_name: str, prob: float
) -> None:
    random.seed(ZR_SEED)

    patch = {"ItemSets": defaultdict(dict)}

    for str_is_id, itemset in drops["ItemSets"].items():
        for ir_id in itemset["ItemReferenceIDs"]:
            if random.random() < prob:
                patch["ItemSets"][str_is_id][str(ir_id)] = 0

    patch_dir = output_path / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    with open(patch_dir / "drops.json", "w") as w:
        json.dump(patch, w, indent=4)


def generate_config_options(
    tdata_path: Path, output_path: Path
) -> List[Dict[str, Any]]:
    patch_options = {
        "original": [],
        "academy": [tdata_path / "patch" / "1013" / "drops.json"],
    }
    zero_rarity_options = {
        "nozr": 0.0,
        "lozr": 0.1,
        "hizr": 0.5,
    }
    rel_tol_options = {
        "lotol": 0.0001,
        "mitol": 0.001,
        "hitol": 0.1,
    }
    blend_options = {
        "minbl": True,
        "maxbl": False,
    }
    drops_path = tdata_path / "drops.json"

    patch_zr_options = {}
    for patch_opt_name, patch_opt in patch_options.items():
        drops = Drops(drops_path=drops_path, patch_paths=patch_opt)

        for zr_opt_name, zr_opt in zero_rarity_options.items():
            patch_zr_opt_name = f"{patch_opt_name}_{zr_opt_name}"

            generate_zero_rarity_patch(drops, output_path, patch_zr_opt_name, zr_opt)

            patch_zr_options[patch_zr_opt_name] = [
                *patch_opt,
                output_path / patch_zr_opt_name / "drops.json",
            ]

    config_options = []
    for patch_zr_opt_name, patch_zr_opt in patch_zr_options.items():
        for rtol_opt_name, rtol_opt in rel_tol_options.items():
            for blend_opt_name, blend_opt in blend_options.items():
                patch_zr_opt_str = [str(p) for p in patch_zr_opt]
                xdt_suffix = (
                    "1013" if any("1013" in p for p in patch_zr_opt_str) else ""
                )

                full_output_path = (
                    output_path
                    / f"{patch_zr_opt_name}_{rtol_opt_name}_{blend_opt_name}"
                )
                full_output_path.mkdir(parents=True, exist_ok=True)

                config_options.append(
                    {
                        "base": str(drops_path),
                        "xdt": str(tdata_path / f"xdt{xdt_suffix}.json"),
                        "generate-patch": True,
                        "log-level": "DEBUG",
                        "patches": patch_zr_opt_str,
                        "rel-tol": rtol_opt,
                        "blend-using-lowest-chance": blend_opt,
                        "output": str(full_output_path),
                    }
                )

    return config_options


def generate_item_configs(
    config_options: Dict[str, Any],
    other_addition_prob: float = 0.2,
    other_zero_prob: float = 0.5,
    criteria_include_prob: float = 0.5,
) -> List[Dict[str, Any]]:
    random.seed(ITEM_SEED)

    with open(config_options["xdt"]) as r:
        xdt = json.load(r)

    type_id_name_map = {
        0: "Weapon",
        1: "Shirts",
        2: "Pants",
        3: "Shoes",
        4: "Hat",
        5: "Glass",
        6: "Back",
        7: "General",
        9: "Chest",
        10: "Vehicle",
    }
    item_map = {
        (i, d["m_iItemNumber"]): {
            **d,
            **xdt[f"m_p{table}ItemTable"]["m_pItemStringData"][d["m_iItemName"]],
        }
        for i, table in type_id_name_map.items()
        for d in xdt[f"m_p{table}ItemTable"]["m_pItemData"]
    }
    crate_name_order_map = {
        "ETC": 0,
        "Standard": 1,
        "Special": 2,
        "Sooper": 3,
        "Sooper Dooper": 4,
    }
    with open("info.json") as r:
        area_names = {
            o["AreaName"] + ("" if o["ZoneName"] != "The Future" else " (Future)")
            for o in json.load(r)["MapRegions"]
            if o["ZoneName"] != "Training"
        }
    crate_names = {
        d["m_strName"] for (type_id, _), d in item_map.items() if type_id == 9
    }
    drops = Drops(
        drops_path=Path(config_options["base"]),
        patch_paths=[Path(p) for p in config_options["patches"]],
    )

    item_count = random.randint(5, 20)
    mob_count_per_mob = random.randint(1, 4)
    crate_count_per_mob = random.randint(1, 4)
    crate_count_per_ct = random.randint(1, 5)
    crate_counts_per_mob_ct = [random.randint(1, 5) for _ in range(5)]
    crate_counts_per_ct_mob = [random.randint(1, 4) for _ in range(6)]
    event_count_per_mob = random.randint(1, 3)
    event_count_per_crate = random.randint(1, 3)
    racing_count = random.randint(1, 4)
    crate_counts_per_racing = [random.randint(1, 5) for _ in range(5)]
    level_count = random.randint(1, 3)
    egger_count = random.randint(1, 4)
    crate_counts_per_egger = [random.randint(1, 5) for _ in range(5)]
    area_count = random.randint(1, 4)
    named_count = random.randint(1, 4)

    items = list(item_map)
    random.shuffle(items)
    item_names = [(*tpl, item_map[tpl]["m_strName"]) for tpl in items[:item_count]]

    mob_ids = list(drops["Mobs"].ikeys())
    random.shuffle(mob_ids)
    mob_index = 0

    event_ids = list(drops["Events"].ikeys())
    random.shuffle(event_ids)
    event_index = 0

    racing_ids = list(drops["Racing"].ikeys())
    random.shuffle(racing_ids)
    racing_index = 0

    levels = list(range(1, 37))
    random.shuffle(levels)
    level_index = 0

    eggers = ["Future", "Suburbs", "Downtown", "Wilds", "Darklands"]
    random.shuffle(eggers)
    egger_index = 0

    areas = list(area_names)
    random.shuffle(areas)
    area_index = 0

    named_crates = list(crate_names)
    random.shuffle(named_crates)
    named_index = 0

    crate_types = []
    for _ in range(100):
        crate_type_list = list(crate_name_order_map.keys())
        random.shuffle(crate_type_list)
        crate_types.append(crate_type_list)
    crate_type_index = 0

    other_keywords = list(OTHER_KEYWORDS)
    other_index = 0

    item_configs = []

    def generate_value() -> int:
        return int(math.exp(random.randint(1, 21)))

    def add_other_element(
        dct: Dict, other_index_list: List[int] = [other_index]
    ) -> None:
        add_other = random.random() < other_addition_prob
        add_other_with_zero = random.random() < other_zero_prob
        other_val = generate_value()

        if add_other:
            dct[other_keywords[other_index_list[0]]] = (
                0 if add_other_with_zero else other_val
            )
            other_index_list[0] = (other_index_list[0] + 1) % len(other_keywords)

    for type_id, item_id, item_name in item_names:
        item_config = {
            "name": item_name,
            "type": type_id,
            "id": item_id,
        }

        val_kills_till_item = generate_value()
        if random.random() < criteria_include_prob:
            item_config["kills-till-item"] = val_kills_till_item

        val_kills_till_item_per_mob = {}
        for _ in range(mob_count_per_mob):
            val_kills_till_item_per_mob[mob_ids[mob_index]] = generate_value()
            mob_index = (mob_index + 1) % len(mob_ids)
        add_other_element(val_kills_till_item_per_mob)
        if random.random() < criteria_include_prob:
            item_config["kills-till-item-per-mob"] = val_kills_till_item_per_mob

        val_crates_till_item = generate_value()
        if random.random() < criteria_include_prob:
            item_config["crates-till-item"] = val_crates_till_item

        val_crates_till_item_per_mob = {}
        for _ in range(crate_count_per_mob):
            val_crates_till_item_per_mob[mob_ids[mob_index]] = generate_value()
            mob_index = (mob_index + 1) % len(mob_ids)
        add_other_element(val_crates_till_item_per_mob)
        if random.random() < criteria_include_prob:
            item_config["crates-till-item-per-mob"] = val_crates_till_item_per_mob

        val_crates_till_item_per_crate_type = {}
        for i in range(crate_count_per_ct):
            val_crates_till_item_per_crate_type[
                crate_types[crate_type_index][i]
            ] = generate_value()
        crate_type_index = (crate_type_index + 1) % len(crate_types)
        add_other_element(val_crates_till_item_per_crate_type)
        if random.random() < criteria_include_prob:
            item_config[
                "crates-till-item-per-crate-type"
            ] = val_crates_till_item_per_crate_type

        val_crates_till_item_per_mob_and_crate_type = {}
        for j, crate_count in enumerate(crate_counts_per_mob_ct[:crate_count_per_mob]):
            if j < crate_count_per_mob - 1:
                mob_id = mob_ids[mob_index]
                mob_index = (mob_index + 1) % len(mob_ids)
            else:
                mob_id = other_keywords[other_index]
                other_index = (other_index + 1) % len(other_keywords)

            val_crates_till_item_per_mob_and_crate_type[mob_id] = {}

            for i in range(crate_count):
                val_crates_till_item_per_mob_and_crate_type[mob_id][
                    crate_types[crate_type_index][i]
                ] = generate_value()
            crate_type_index = (crate_type_index + 1) % len(crate_types)
            add_other_element(val_crates_till_item_per_mob_and_crate_type[mob_id])
        if random.random() < criteria_include_prob:
            item_config[
                "crates-till-item-per-mob-and-crate-type"
            ] = val_crates_till_item_per_mob_and_crate_type

        val_crates_till_item_per_crate_type_and_mob = {}
        current_crate_types = crate_types[crate_type_index]
        crate_type_index = (crate_type_index + 1) % len(crate_types)
        for j, crate_count in enumerate(crate_counts_per_ct_mob[:crate_count_per_ct]):
            if j < crate_count_per_ct - 1:
                crate_type = current_crate_types[j]
            else:
                crate_type = other_keywords[other_index]
                other_index = (other_index + 1) % len(other_keywords)

            val_crates_till_item_per_crate_type_and_mob[crate_type] = {}

            for i in range(crate_count):
                val_crates_till_item_per_crate_type_and_mob[crate_type][
                    mob_ids[mob_index]
                ] = generate_value()
                mob_index = (mob_index + 1) % len(mob_ids)
            add_other_element(val_crates_till_item_per_crate_type_and_mob[crate_type])
        if random.random() < criteria_include_prob:
            item_config[
                "crates-till-item-per-crate-type-and-mob"
            ] = val_crates_till_item_per_crate_type_and_mob

        val_event_kills_till_item = {}
        for _ in range(event_count_per_mob):
            val_event_kills_till_item[event_ids[event_index]] = generate_value()
            event_index = (event_index + 1) % len(event_ids)
        add_other_element(val_event_kills_till_item)
        event_index = 0
        if random.random() < criteria_include_prob:
            item_config["event-kills-till-item"] = val_event_kills_till_item

        val_event_crates_till_item = {}
        for _ in range(event_count_per_crate):
            val_event_crates_till_item[event_ids[event_index]] = generate_value()
            event_index = (event_index + 1) % len(event_ids)
        add_other_element(val_event_crates_till_item)
        event_index = 0
        if random.random() < criteria_include_prob:
            item_config["event-crates-till-item"] = val_event_crates_till_item

        val_race_crates_till_item = {}
        for j, crate_count in enumerate(crate_counts_per_racing[:racing_count]):
            if j < racing_count - 1:
                racing_id = racing_ids[racing_index]
                racing_index = (racing_index + 1) % len(racing_ids)
            else:
                racing_id = other_keywords[other_index]
                other_index = (other_index + 1) % len(other_keywords)

            val_race_crates_till_item[racing_id] = {}

            for i in range(crate_count):
                val_race_crates_till_item[racing_id][
                    crate_types[crate_type_index][i]
                ] = generate_value()
            crate_type_index = (crate_type_index + 1) % len(crate_types)
            add_other_element(val_race_crates_till_item[racing_id])
        if random.random() < criteria_include_prob:
            item_config["race-crates-till-item"] = val_race_crates_till_item

        val_mission_crates_till_item = {}
        for _ in range(level_count):
            val_mission_crates_till_item[levels[level_index]] = generate_value()
            level_index = (level_index + 1) % len(levels)
        add_other_element(val_mission_crates_till_item)
        if random.random() < criteria_include_prob:
            item_config["mission-crates-till-item"] = val_mission_crates_till_item

        val_egger_crates_till_item = {}
        for j, crate_count in enumerate(crate_counts_per_egger[:egger_count]):
            if j < egger_count - 1:
                egger_zone = eggers[egger_index]
                egger_index = (egger_index + 1) % len(eggers)
            else:
                egger_zone = other_keywords[other_index]
                other_index = (other_index + 1) % len(other_keywords)

            val_egger_crates_till_item[egger_zone] = {}

            for i in range(crate_count):
                val_egger_crates_till_item[egger_zone][
                    crate_types[crate_type_index][i]
                ] = generate_value()
            crate_type_index = (crate_type_index + 1) % len(crate_types)
            add_other_element(val_egger_crates_till_item[egger_zone])
        if random.random() < criteria_include_prob:
            item_config["egger-crates-till-item"] = val_egger_crates_till_item

        val_golden_eggs_till_item = {}
        for _ in range(area_count):
            val_golden_eggs_till_item[areas[area_index]] = generate_value()
            area_index = (area_index + 1) % len(areas)
        add_other_element(val_golden_eggs_till_item)
        if random.random() < criteria_include_prob:
            item_config["golden-eggs-till-item"] = val_golden_eggs_till_item

        val_named_crates_till_item = {}
        for _ in range(named_count):
            val_named_crates_till_item[named_crates[named_index]] = generate_value()
            named_index = (named_index + 1) % len(named_crates)
        add_other_element(val_named_crates_till_item)
        if random.random() < criteria_include_prob:
            item_config["named-crates-till-item"] = val_named_crates_till_item

        item_configs.append(item_config)

    return item_configs


def test() -> None:
    parser = ArgumentParser("Rarify test program.")
    parser.add_argument("--tdata", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config_options_list = generate_config_options(
        tdata_path=Path(args.tdata), output_path=Path(args.output)
    )

    full_config_options_list = [
        {**config_options, "items": generate_item_configs(config_options)}
        for config_options in config_options_list
    ]

    for full_config_options in full_config_options_list:
        output_path = Path(full_config_options["output"])

        with open("cfg.yml", "w") as w:
            yaml.safe_dump(full_config_options, w, sort_keys=False, indent=2)

        print("Running for output path:", output_path)

        completed_process = subprocess.run(
            ["python", "main.py", "cfg.yml"],
            encoding="utf-8",
        )
        Path("cfg.yml").replace(output_path / "cfg.yml")

        if completed_process.returncode != 0:
            raise AssertionError("Process did not complete properly.")

        print("Ran successfully. Moving files ...")
        Path("rarify.log").replace(output_path / "rarify.log")


if __name__ == "__main__":
    test()
