import json
from typing import Any, Dict, List
from collections import defaultdict

from rarify.config import Config, sanitize_key
from rarify.drops import Drops, get_patched


class KnowledgeBase:
    def __init__(self, config: Config) -> None:
        with open("info.json") as r:
            self.info = json.load(r)

        self.gender_map: Dict[str, int] = self.info["GenderMap"]
        self.rarity_map: Dict[str, int] = self.info["RarityMap"]

        self.crate_type_name_to_crate_type_map: Dict[str, str] = self.info[
            "CrateTypeNameToCrateType"
        ]
        self.crate_type_to_crate_order_map: Dict[str, int] = self.info[
            "CrateTypeToCrateOrder"
        ]
        self.crate_order_to_crate_type_map = {
            o: n for n, o in self.crate_type_to_crate_order_map.items()
        }

        self.item_type_name_to_item_type_map: Dict[str, str] = self.info[
            "ItemTypeNameToItemType"
        ]
        self.item_type_to_item_type_id_map: Dict[str, int] = self.info[
            "ItemTypeToItemTypeID"
        ]
        self.item_type_id_to_item_type_map = {
            i: t for t, i in self.item_type_to_item_type_id_map.items()
        }

        self.event_name_to_event_type_map: Dict[str, str] = self.info[
            "EventNameToEventType"
        ]
        self.event_type_to_event_id_map: Dict[str, int] = self.info[
            "EventTypeToEventID"
        ]
        self.event_id_to_event_type_map = {
            i: e for e, i in self.event_type_to_event_id_map.items()
        }

        self.zone_name_to_start_egger_crate_id: Dict[str, int] = self.info[
            "ZoneNameToStartEggerCrateID"
        ]
        self.zone_to_egger_map = {
            zone_name: {
                crate_type: start_id + crate_order
                for crate_type, crate_order in self.crate_type_to_crate_order_map.items()
                if crate_type != "ETC"
            }
            for zone_name, start_id in self.zone_name_to_start_egger_crate_id.items()
        }
        self.zone_to_mystery_egger_map = {
            zone_name: {
                crate_type: crate_id + 4 for crate_type, crate_id in egger_map.items()
            }
            for zone_name, egger_map in self.zone_to_egger_map.items()
        }

        self.map_regions: List[Dict[str, Any]] = self.info["MapRegions"]
        self.name_to_areas = defaultdict(list)
        for obj in self.map_regions:
            area_key = sanitize_key(obj["AreaName"]).replace("the ", "")
            if "Future" in obj["ZoneName"]:
                area_key += " f"
            self.name_to_areas[area_key].append(obj)

        self.base_drops = Drops(config.base, config.patches)
        self.drops = Drops(config.base, config.patches)
        self.mobs = get_patched(
            base=(config.base.parent / "mobs.json"),
            patches=[
                patch_path.parent / "mobs.json"
                for patch_path in config.patches
                if (patch_path.parent / "mobs.json").is_file()
            ],
        )
        self.eggs = get_patched(
            base=(config.base.parent / "eggs.json"),
            patches=[
                patch_path.parent / "eggs.json"
                for patch_path in config.patches
                if (patch_path.parent / "eggs.json").is_file()
            ],
        )
        with open(config.xdt) as r:
            self.xdt = json.load(r)

        self.egg_type_to_crate_id_map = {
            egg["Id"]: egg["DropCrateId"] for egg in self.eggs["EggTypes"].values()
        }

        self.area_eggs = defaultdict(set)
        for egg in self.eggs["Eggs"].values():
            crate_id = self.egg_type_to_crate_id_map.get(egg["iType"])
            if not crate_id:
                continue

            found = False
            for area_name, areas in self.name_to_areas.items():
                for area in areas:
                    if (area["X"] <= egg["iX"] < area["X"] + area["Width"]) and (
                        area["Y"] <= egg["iY"] < area["Y"] + area["Height"]
                    ):
                        self.area_eggs[area_name].add(crate_id)
                        found = True
                        break
                if found:
                    break
        self.area_eggs = {k: list(v) for k, v in self.area_eggs.items()}

        self.loaded_mobs = {mob["iNPCType"] for mob in self.mobs["mobs"].values()}
        for mob_group in self.mobs["groups"].values():
            self.loaded_mobs.add(mob_group["iNPCType"])
            for mob in mob_group["aFollowers"]:
                self.loaded_mobs.add(mob["iNPCType"])

        self.iz_name_to_epid_map = {
            sanitize_key(obj["EPName"]).replace("the ", ""): epid
            for epid, obj in self.base_drops["Racing"].iitems()
        }
        self.iz_name_to_epid_map.update(self.info["IZNameToEPID"])

        self.item_map = {
            (i, d["m_iItemNumber"]): {
                **d,
                **self.xdt[f"m_p{table}ItemTable"]["m_pItemStringData"][
                    d["m_iItemName"]
                ],
            }
            for i, table in self.item_type_id_to_item_type_map.items()
            for d in self.xdt[f"m_p{table}ItemTable"]["m_pItemData"]
        }

        self.item_name_to_tuples_map = defaultdict(set)
        for item_tuple, d in self.item_map.items():
            name_sanitized = sanitize_key(d["m_strName"])
            self.item_name_to_tuples_map[name_sanitized].add(item_tuple)
        self.item_name_to_tuples_map = {
            k: list(v) for k, v in self.item_name_to_tuples_map.items()
        }

        self.mission_level_crate_id_map = defaultdict(set)
        for item_name, item_tuples in self.item_name_to_tuples_map.items():
            if "mission crate" in item_name:
                level = int(item_name.split("lv")[0])
                for item_type, crate_id in item_tuples:
                    if item_type == 9 and crate_id in self.drops["Crates"]:
                        self.mission_level_crate_id_map[level].add(crate_id)
        self.mission_level_crate_id_map = {
            k: list(v) for k, v in self.mission_level_crate_id_map.items() if v
        }

        self.npc_map = {
            d["m_iNpcNumber"]: {
                **d,
                "m_strName": self.xdt["m_pNpcTable"]["m_pNpcStringData"][
                    d["m_iNpcName"]
                ]["m_strName"],
            }
            for d in self.xdt["m_pNpcTable"]["m_pNpcData"]
        }

        self.npc_name_to_npc_ids_map = defaultdict(set)
        for npc_id, d in self.npc_map.items():
            name_sanitized = sanitize_key(d["m_strName"])
            self.npc_name_to_npc_ids_map[name_sanitized].add(npc_id)
        self.npc_name_to_npc_ids_map = {
            k: list(v) for k, v in self.npc_name_to_npc_ids_map.items()
        }

        self.tuple_to_ir_id_map = {
            (d["Type"], d["ItemID"]): d["ItemReferenceID"]
            for d in self.base_drops["ItemReferences"].values()
        }
        self.ir_id_to_tuple_map = {
            d["ItemReferenceID"]: (d["Type"], d["ItemID"])
            for d in self.base_drops["ItemReferences"].values()
        }
