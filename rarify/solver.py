import math
import json
import logging
from operator import itemgetter
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Set

from .drops import Data
from .knowledge_base import KnowledgeBase
from .config import ItemConfig, OTHER_STANDARD_ID, OTHER_STANDARD_KEYWORD

ABS_TOL = 0.01


def inv(a: float) -> float:
    return 1.0 / a if a > 0.0 else float("inf")


def abs_diff(a: float, b: float) -> float:
    if a == float("inf") and b == float("inf"):
        return 0.0
    return abs(a - b)


class ItemSetNode:
    def __init__(self, knowledge_base: KnowledgeBase, is_id: int) -> None:
        self.knowledge_base = knowledge_base
        self.is_id = is_id
        self.itemset: Data = knowledge_base.drops["ItemSets"][is_id]
        self.probs_to_change = {
            gender_id: {rarity_id: {} for rarity_id in range(1, 5)}
            for gender_id in range(1, 3)
        }

    def effective_rarity_id(self, ir_id: int) -> int:
        if self.itemset["IgnoreRarity"]:
            return 0
        return self.itemset["AlterRarityMap"].get(
            str(ir_id),
            self.knowledge_base.item_map[self.knowledge_base.irid_tuple_map[ir_id]].get(
                "m_iRarity", 0
            ),
        )

    def effective_gender_id(self, ir_id: int) -> int:
        if self.itemset["IgnoreGender"]:
            return 0
        return self.itemset["AlterGenderMap"].get(
            str(ir_id),
            self.knowledge_base.item_map[self.knowledge_base.irid_tuple_map[ir_id]].get(
                "m_iReqSex", 0
            ),
        )

    def register(self, ir_id: int, changed_rarity_id: int) -> None:
        if ir_id not in self.itemset["ItemReferenceIDs"]:
            self.itemset["ItemReferenceIDs"].append(ir_id)
            self.itemset["AlterItemWeightMap"][str(ir_id)] = 0

            rarity_id = self.effective_rarity_id(ir_id)
            if rarity_id != 0 and changed_rarity_id != rarity_id:
                self.itemset["AlterRarityMap"][str(ir_id)] = changed_rarity_id

    def drop_pool(
        self,
        rarity_id: int,
        gender_id: int,
        target_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        weights_filtered = {
            ir_id: self.itemset["AlterItemWeightMap"].get(
                str(ir_id), self.itemset["DefaultItemWeight"]
            )
            for ir_id in self.itemset["ItemReferenceIDs"]
            if (
                self.effective_gender_id(ir_id) in [0, gender_id]
                and self.effective_rarity_id(ir_id) in [0, rarity_id]
            )
        }
        total_weight = sum(weights_filtered.values())
        weights_scaled = {
            ir_id: weight / total_weight for ir_id, weight in weights_filtered.items()
        }

        if not target_weights:
            return weights_scaled

        target_weights_filtered = {
            ir_id: weight
            for ir_id, weight in target_weights.items()
            if ir_id in weights_scaled
        }
        total_target_weight = sum(target_weights_filtered.values())
        total_non_target_weight = sum(
            [
                w
                for ir_id, w in weights_scaled.items()
                if ir_id not in target_weights_filtered
            ]
        )

        # Assumption: ItemReferenceIDs is updated and contains all that it has to contain
        return {
            ir_id: (
                target_weights_filtered[ir_id]
                if ir_id in target_weights_filtered
                else (1.0 - total_target_weight) * weight / total_non_target_weight
            )
            for ir_id, weight in weights_scaled.items()
        }

    def inject(self, ir_id: int, prob: float, agg: Callable = min) -> None:
        item_gender_id = self.effective_gender_id(ir_id)
        item_rarity_id = self.effective_rarity_id(ir_id)

        for gender_id, rarity_dicts in self.probs_to_change.items():
            for rarity_id, pool_dict in rarity_dicts.items():
                if item_gender_id in [0, gender_id] and item_rarity_id in [
                    0,
                    rarity_id,
                ]:
                    other_probs_sum = sum(
                        [
                            value
                            for other_ir_id, value in pool_dict.items()
                            if other_ir_id != ir_id
                        ]
                    )

                    if other_probs_sum + prob > 1.0:
                        logging.warning(
                            "ItemSet %s rejected probability %s for ItemReference %s "
                            "as %s%s exceeds 1.0",
                            self.is_id,
                            prob,
                            ir_id,
                            (
                                f"previously saved total probability {other_probs_sum} + "
                                if other_probs_sum > 0
                                else ""
                            ),
                            prob,
                        )
                        continue

                    pool_dict[ir_id] = agg([pool_dict.get(ir_id, prob), prob])

    def remove_zero_prob_entries(
        self,
        merged_probs: Dict[int, Dict[int, Dict[int, float]]],
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        zero_entries = {
            ir_id
            for ir_id in self.itemset["ItemReferenceIDs"]
            if all(
                prob_dict[ir_id] == 0.0
                for rarity_dict in merged_probs.values()
                for prob_dict in rarity_dict.values()
                if ir_id in prob_dict
            )
        }

        return {
            gender_id: {
                rarity_id: {
                    ir_id: value
                    for ir_id, value in prob_dict.items()
                    if ir_id not in zero_entries
                }
                for rarity_id, prob_dict in rarity_dict.items()
            }
            for gender_id, rarity_dict in merged_probs.items()
        }

    def get_scaled_int_weights(
        self,
        merged_probs: Dict[int, Dict[int, Dict[int, float]]],
        agg: Callable = min,
    ) -> Dict[int, Dict[int, Dict[int, int]]]:
        def apply_scale(prob_dict: Dict[int, float], scale: int) -> Dict[int, int]:
            return {ir_id: int(scale * value) for ir_id, value in prob_dict.items()}

        def search(
            prob_dict: Dict[int, float], lo: int = 1, hi: int = (1 << 31) - 1
        ) -> int:
            if hi < lo:
                return -1

            mi = (lo + hi) // 2

            current_values = apply_scale(prob_dict, mi)
            cur_sum = sum(current_values.values())

            if all(
                abs_diff(cur_sum * inv(value), inv(prob_dict[ir_id])) < ABS_TOL
                for ir_id, value in current_values.items()
            ):
                left = search(prob_dict, lo, mi - 1)
                return mi if left == -1 else left

            return search(prob_dict, mi + 1, hi)

        def merge_by_rarity(
            acc_dict: Dict[int, int],
            other_dict: Dict[int, int],
        ) -> Tuple[int, Dict[int, int]]:
            shared_ir_ids = list(set(acc_dict.keys()).intersection(other_dict))

            if not shared_ir_ids:
                return 1, {**acc_dict, **other_dict}

            select_ir_id = shared_ir_ids[0]
            scale_gcd = math.gcd(acc_dict[select_ir_id], other_dict[select_ir_id])
            other_to_scale = acc_dict[select_ir_id] // scale_gcd
            acc_to_scale = other_dict[select_ir_id] // scale_gcd

            return acc_to_scale, {
                **{ir_id: value * acc_to_scale for ir_id, value in acc_dict.items()},
                **{
                    ir_id: value * other_to_scale for ir_id, value in other_dict.items()
                },
            }

        scales = {}
        weights = {}

        for gender_id, rarity_dict in merged_probs.items():
            scales[gender_id] = 1
            weights[gender_id] = {}

            for prob_dict in rarity_dict.values():
                scale = search(prob_dict)
                int_weights = apply_scale(prob_dict, scale)

                multp, weights[gender_id] = merge_by_rarity(
                    weights[gender_id], int_weights
                )
                # this is a heuristic, but we have to do something meaningless here to decide
                scales[gender_id] = max(scales[gender_id] * multp, scale)

        # where the true values of the gender are kept
        main_gender_id, _ = agg(list(scales.items()), key=itemgetter(1))
        # where all other items are dumped from
        other_gender_id = 1 if main_gender_id == 2 else 2

        return {
            **weights[other_gender_id],
            **{
                ir_id: value
                for ir_id, value in weights[main_gender_id].items()
                if self.effective_gender_id(ir_id) == main_gender_id
            },
        }

    def adjust_weight_settings(self, weights: Dict[int, int]) -> None:
        counts = defaultdict(int)
        for weight in weights.values():
            counts[weight] += 1

        existing_ids = dict.fromkeys(self.itemset["ItemReferenceIDs"])
        entry_list = []

        for ir_id in existing_ids:
            if ir_id in weights:
                entry_list.append((ir_id, weights[ir_id]))

        for ir_id, weight in sorted(weights.items()):
            if ir_id not in existing_ids:
                entry_list.append((ir_id, weight))

        self.itemset["DefaultItemWeight"] = max(counts.items(), key=itemgetter(1))[0]
        self.itemset["ItemReferenceIDs"] = [ir_id for ir_id, _ in entry_list]
        self.itemset["AlterItemWeightMap"] = {
            str(ir_id): weight
            for ir_id, weight in entry_list
            if weight != self.itemset["DefaultItemWeight"]
        }

    def alter_drops_values(self, agg: Callable = min) -> None:
        if not any(
            len(pool_dict) > 0
            for rarity_dicts in self.probs_to_change.values()
            for pool_dict in rarity_dicts.values()
        ):
            return

        unchanged_probs = {
            gender_id: {
                rarity_id: self.drop_pool(
                    rarity_id=rarity_id,
                    gender_id=gender_id,
                )
                for rarity_id in rarity_dicts
            }
            for gender_id, rarity_dicts in self.probs_to_change.items()
        }

        logging.debug(
            "ItemSet %s current pools %s",
            self.is_id,
            json.dumps(unchanged_probs, indent=4),
        )

        merged_probs = {
            gender_id: {
                rarity_id: self.drop_pool(
                    rarity_id=rarity_id,
                    gender_id=gender_id,
                    target_weights=pool_dict,
                )
                for rarity_id, pool_dict in rarity_dicts.items()
            }
            for gender_id, rarity_dicts in self.probs_to_change.items()
        }

        logging.debug(
            "ItemSet %s probs to change %s",
            self.is_id,
            json.dumps(self.probs_to_change, indent=4),
        )

        logging.debug(
            "ItemSet %s changed pools %s",
            self.is_id,
            json.dumps(merged_probs, indent=4),
        )

        clean_merged_probs = self.remove_zero_prob_entries(merged_probs)

        logging.debug(
            "ItemSet %s cleaned pools %s",
            self.is_id,
            json.dumps(clean_merged_probs, indent=4),
        )

        int_weights = self.get_scaled_int_weights(clean_merged_probs, agg=agg)

        logging.debug(
            "ItemSet %s int weights %s", self.is_id, json.dumps(int_weights, indent=4)
        )

        self.adjust_weight_settings(int_weights)


class CrateNode:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        itemset_register: Dict[int, ItemSetNode],
        c_id: int,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.itemset_register = itemset_register
        self.c_id = c_id

        self.crate = knowledge_base.drops["Crates"][c_id]
        self.rarity_weights = knowledge_base.drops["RarityWeights"][
            self.crate["RarityWeightID"]
        ]

        is_id = self.crate["ItemSetID"]
        if is_id not in self.itemset_register:
            self.itemset_register[is_id] = ItemSetNode(knowledge_base, is_id)

        self.itemset_node = itemset_register[is_id]

    def allowed_rarities(self, gender_id: int) -> Dict[int, int]:
        return {
            rarity_id: weight
            for rarity_id, weight in zip(range(1, 5), self.rarity_weights["Weights"])
            if (
                weight > 0
                and any(
                    w > 0
                    for w in self.itemset_node.drop_pool(
                        gender_id=gender_id,
                        rarity_id=rarity_id,
                    ).values()
                )
            )
        }

    def register(self, ir_id: int) -> None:
        item_tuple = self.knowledge_base.irid_tuple_map[ir_id]
        item_gender_id = self.knowledge_base.item_map[item_tuple].get("m_iReqSex", 1)
        item_rarity_id = self.knowledge_base.item_map[item_tuple].get("m_iRarity", 1)

        changed_rarity = max(
            rarity_id
            for rarity_id in self.allowed_rarities(item_gender_id)
            if rarity_id <= item_rarity_id
        )
        self.itemset_node.register(ir_id=ir_id, changed_rarity_id=changed_rarity)

    def discount_explained_prob(self, ir_id: int, prob: float) -> float:
        rarity_id = self.itemset_node.effective_rarity_id(ir_id)

        if rarity_id == 0:
            logging.debug(
                "Item %s probability %s is 0 rarity from crate %s to item set %s",
                ir_id,
                prob,
                self.c_id,
                self.crate["ItemSetID"],
            )
            # TODO: not sure about this one...
            discounted_prob = prob
        else:
            gender_id = max(1, self.itemset_node.effective_gender_id(ir_id))
            rw_weights = self.rarity_weights["Weights"]
            rw_sum = sum(self.allowed_rarities(gender_id).values())

            if len(rw_weights) < rarity_id or rw_weights[rarity_id - 1] == 0:
                discounted_prob = 0.0
                logging.debug(
                    "Item %s probability %s is 0 due to rarity_id %s and weights %s from crate %s to item set %s",
                    ir_id,
                    prob,
                    rarity_id,
                    rw_weights,
                    self.c_id,
                    self.crate["ItemSetID"],
                )
            else:
                discounted_prob = prob * rw_sum / rw_weights[rarity_id - 1]

        logging.debug(
            "Item %s probability %s discounted to %s from crate %s to item set %s",
            ir_id,
            prob,
            discounted_prob,
            self.c_id,
            self.crate["ItemSetID"],
        )

        return discounted_prob

    def drop_pool(
        self,
        gender_id: int,
        target_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        for ir_id in target_weights or {}:
            self.register(ir_id)

        rarity_agg = defaultdict(float)

        rw_weights = self.rarity_weights["Weights"]
        rw_sum = sum(self.allowed_rarities(gender_id).values())

        for rarity_id, rw in zip(range(1, 5), rw_weights):
            pool = self.itemset_node.drop_pool(
                rarity_id=rarity_id,
                gender_id=gender_id,
                target_weights=(
                    None
                    if target_weights is None
                    else {
                        ir_id: self.discount_explained_prob(ir_id, weight)
                        for ir_id, weight in target_weights.items()
                    }
                ),
            )
            for ir_id, weight in pool.items():
                rarity_agg[ir_id] += weight * rw / rw_sum

        return rarity_agg

    def inject(self, ir_id: int, prob: float, agg: Callable = min) -> None:
        self.register(ir_id)
        self.itemset_node.inject(
            ir_id=ir_id,
            prob=self.discount_explained_prob(ir_id, prob),
            agg=agg,
        )


class CrateGroupNode:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        itemset_register: Dict[int, ItemSetNode],
        crate_register: Dict[int, CrateNode],
        cdc_id: int,
        cdt_id: int,
        is_id: int,
        agg: Callable = min,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.itemset_register = itemset_register
        self.crate_register = crate_register
        self.cdc_id = cdc_id
        self.cdt_id = cdt_id
        self.is_id = is_id
        self.agg = agg

        self.cdc = knowledge_base.drops["CrateDropChances"][cdc_id]
        self.cdt = knowledge_base.drops["CrateDropTypes"][cdt_id]

        if is_id not in self.itemset_register:
            self.itemset_register[is_id] = ItemSetNode(knowledge_base, is_id)
        self.itemset_node = itemset_register[is_id]

        for crate_id in self.cdt["CrateIDs"]:
            if crate_id not in self.crate_register:
                self.crate_register[crate_id] = CrateNode(
                    knowledge_base, itemset_register, crate_id
                )
        self.crate_nodes = {
            index: self.crate_register[crate_id]
            for index, crate_id in enumerate(self.cdt["CrateIDs"])
            if self.is_id == self.crate_register[crate_id].itemset_node.is_id
        }

    def discount_explained_prob(
        self,
        crate_index: int,
        prob: float,
        ignore_any_crate_prob: bool = False,
    ) -> float:
        any_crate_prob = self.cdc["DropChance"] / self.cdc["DropChanceTotal"]
        cdw = self.cdc["CrateTypeDropWeights"]
        cdwt = sum(cdw)

        if len(cdw) < crate_index + 1 or cdw[crate_index] == 0:
            discounted_prob = 0.0
        else:
            discounted_prob = (
                prob
                * cdwt
                / (
                    cdw[crate_index]
                    * (1.0 if ignore_any_crate_prob else any_crate_prob)
                )
            )

        logging.debug(
            "Probability %s discounted to %s from crate group %s to crate %s",
            prob,
            discounted_prob,
            (self.cdc_id, self.cdt_id, self.is_id),
            self.cdt["CrateIDs"][crate_index],
        )

        return discounted_prob

    def drop_pool(
        self,
        gender_id: int,
        target_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        any_crate_prob = self.cdc["DropChance"] / self.cdc["DropChanceTotal"]
        cdw = self.cdc["CrateTypeDropWeights"]
        cdwt = sum(cdw)
        group_agg = defaultdict(float)

        for index, crate_node in self.crate_nodes.items():
            pool = crate_node.drop_pool(
                gender_id=gender_id,
                target_weights=(
                    None
                    if target_weights is None
                    else {
                        ir_id: self.discount_explained_prob(index, weight)
                        for ir_id, weight in target_weights.items()
                    }
                ),
            )
            for ir_id, weight in pool.items():
                group_agg[ir_id] += any_crate_prob * cdw[index] * weight / cdwt

        return group_agg

    def generate_splits(
        self,
        ir_id: int,
        crate_target_probs: Dict[int, float],
    ) -> List[Tuple["CrateGroupNode", int, float]]:
        max_index = len(self.cdt["CrateIDs"]) - 1
        fixed_crate_target_probs = {}

        for index, prob in sorted(crate_target_probs.items()):
            fixed_index = min(max_index, index)

            if fixed_index in fixed_crate_target_probs:
                logging.warn(
                    "Could not truncate index %s for CrateIDs in CrateDropType %s to "
                    "%s, as this index has valid prior configuration, skipping ...",
                    index,
                    self.cdt_id,
                    fixed_index,
                )
            else:
                fixed_crate_target_probs[fixed_index] = prob

        # this may not work for rarity_id=0
        item_freq = inv(
            self.agg(
                [
                    weights[ir_id]
                    for gender_id in self.itemset_node.probs_to_change
                    for weights in [self.drop_pool(gender_id=gender_id)]
                    if ir_id in weights
                ]
            )
        )

        freq_groups = defaultdict(list)
        # stable sort strats
        freq_groups[item_freq].extend(self.crate_nodes.keys())

        for index, crate_node in self.crate_nodes.items():
            if index not in fixed_crate_target_probs:
                continue

            desired_freq = inv(
                crate_node.discount_explained_prob(
                    ir_id, fixed_crate_target_probs[index]
                )
            )

            added = False
            for freq, freq_group in freq_groups.items():
                if abs_diff(freq, desired_freq) < ABS_TOL:
                    freq_group.append(index)
                    added = True
                    break

            if not added:
                freq_groups[desired_freq].append(index)
            freq_groups[item_freq].remove(index)

        if not freq_groups[item_freq]:
            del freq_groups[item_freq]

        # in the case of a tie, this should still leave the original group at the top
        sorted_freq_groups = sorted(
            freq_groups.items(),
            key=lambda t: len(t[1]),
            reverse=True,
        )

        groups = []
        for i, (freq, freq_group) in enumerate(sorted_freq_groups):
            if i == 0:
                is_id = self.is_id
            else:
                new_itemset = self.itemset_node.itemset.deepcopy()
                is_id = self.knowledge_base.drops["ItemSets"].add(new_itemset)

            for index in freq_group:
                crate_id = self.cdt["CrateIDs"][index]
                crate = self.knowledge_base.drops["Crates"][crate_id]
                # this should also fix the references
                crate["ItemSetID"] = is_id

                # force reload
                self.crate_register[crate_id] = CrateNode(
                    knowledge_base=self.knowledge_base,
                    itemset_register=self.itemset_register,
                    c_id=crate_id,
                )

            groups.append((is_id, freq_group[0], 1.0 / freq))

        # CrateGroupNode construction has to happen after ALL crate_register entries are changed
        return [
            (
                CrateGroupNode(
                    knowledge_base=self.knowledge_base,
                    itemset_register=self.itemset_register,
                    crate_register=self.crate_register,
                    cdc_id=self.cdc_id,
                    cdt_id=self.cdt_id,
                    is_id=is_id,
                    agg=self.agg,
                ),
                main_index,
                prob,
            )
            for is_id, main_index, prob in groups
        ]

    def inject(
        self,
        ir_id: int,
        prob: float,
        ignore_any_crate_prob: bool = False,
        agg: Callable = min,
    ) -> None:
        for index, crate_node in self.crate_nodes.items():
            # TODO: count satisfied % and distribute remaining
            crate_node.inject(
                ir_id=ir_id,
                prob=self.discount_explained_prob(
                    crate_index=index,
                    prob=prob,
                    ignore_any_crate_prob=ignore_any_crate_prob,
                ),
                agg=agg,
            )


class ConfigKnowledgeBase:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        item_configs: List[ItemConfig],
        agg: Callable = min,
    ) -> None:
        self.knowledge_base = knowledge_base

        references = knowledge_base.drops.references
        self.ir_is_ids = {
            self.get_ir_id(item_config): {
                is_id
                for map_name_is, is_id in references.get(
                    ("ItemReferences", self.get_ir_id(item_config)), []
                )
                if map_name_is == "ItemSets"
            }
            for item_config in item_configs
        }
        self.itemset_register = {
            is_id: ItemSetNode(knowledge_base, is_id)
            for is_ids in self.ir_is_ids.values()
            for is_id in is_ids
        }

        self.ir_c_ids = {
            ir_id: {
                c_id
                for is_id in is_ids
                for map_name_c, c_id in references.get(("ItemSets", is_id), [])
                if map_name_c == "Crates"
            }
            for ir_id, is_ids in self.ir_is_ids.items()
        }
        self.crate_register = {
            c_id: CrateNode(knowledge_base, self.itemset_register, c_id)
            # pull them all, we need to recognize any and all crates
            for c_id in knowledge_base.drops["Crates"]
        }

        self.ir_triples = defaultdict(set)
        self.crate_drop_m_ids = defaultdict(set)
        self.crate_drop_e_ids = defaultdict(set)

        for ir_id, c_ids in self.ir_c_ids.items():
            for c_id in c_ids:
                crate = knowledge_base.drops["Crates"][c_id]

                for map_name_cdt, cdt_id in references.get(("Crates", c_id), []):
                    if map_name_cdt != "CrateDropTypes":
                        continue

                    for map_name_md, md_id in references.get(
                        ("CrateDropTypes", cdt_id), []
                    ):
                        if map_name_md != "MobDrops":
                            continue

                        mobdrop = knowledge_base.drops["MobDrops"][md_id]
                        tpl = (
                            mobdrop["CrateDropChanceID"],
                            mobdrop["CrateDropTypeID"],
                        )
                        triple = (*tpl, crate["ItemSetID"])
                        self.ir_triples[ir_id].add(triple)

                        for map_name_me, me_id in references.get(
                            ("MobDrops", md_id), []
                        ):
                            if map_name_me == "Mobs":
                                self.crate_drop_m_ids[tpl].add(me_id)
                            elif map_name_me == "Events":
                                self.crate_drop_e_ids[tpl].add(me_id)

        self.crate_group_register = {
            (cdc_id, cdt_id, is_id): CrateGroupNode(
                knowledge_base=knowledge_base,
                itemset_register=self.itemset_register,
                crate_register=self.crate_register,
                cdc_id=cdc_id,
                cdt_id=cdt_id,
                is_id=is_id,
                agg=agg,
            )
            for triples in self.ir_triples.values()
            for cdc_id, cdt_id, is_id in triples
        }

    def get_ir_id(self, item_config: ItemConfig) -> int:
        item_tuple = (item_config.type, item_config.id)
        ir_id = self.knowledge_base.tuple_irid_map.get(item_tuple)

        if item_tuple not in self.knowledge_base.tuple_irid_map:
            ir_id = self.knowledge_base.drops["ItemReferences"].add(
                {
                    "ItemReferenceID": -1,
                    "ItemID": item_config.id,
                    "Type": item_config.type,
                }
            )
            self.knowledge_base.tuple_irid_map[item_tuple] = ir_id
            self.knowledge_base.irid_tuple_map[ir_id] = item_tuple

        return ir_id


def inject_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq: float,
    ignore_any_crate_prob: bool = False,
    agg: Callable = min,
) -> None:
    prob = 1.0 / freq
    for triple in ckb.ir_triples[ir_id]:
        ckb.crate_group_register[triple].inject(
            ir_id=ir_id,
            prob=prob,
            ignore_any_crate_prob=ignore_any_crate_prob,
            agg=agg,
        )


def inject_mob_event_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    map_name: str,
    freq_per_mob_event: Dict[int, float],
    ignore_any_crate_prob: bool = False,
    agg: Callable = min,
) -> None:
    filtered_config = {}
    unspecified_triplets = ckb.ir_triples[ir_id].copy()

    for mob_event_id, freq in freq_per_mob_event.items():
        if mob_event_id == OTHER_STANDARD_ID:
            continue

        mob_event = ckb.knowledge_base.drops[map_name][mob_event_id]
        mob_drop = ckb.knowledge_base.drops["MobDrops"][mob_event["MobDropID"]]

        cdt_id = mob_drop["CrateDropTypeID"]
        cdt = ckb.knowledge_base.drops["CrateDropTypes"][cdt_id]
        found_triples = set()

        for crate_id in cdt["CrateIDs"]:
            crate = ckb.knowledge_base.drops["Crates"][crate_id]
            triple = (mob_drop["CrateDropChanceID"], cdt_id, crate["ItemSetID"])

            if triple in ckb.ir_triples[ir_id]:
                found_triples.add(triple)

        for triple in found_triples:
            value = 1.0 / freq

            if triple in filtered_config:
                old_value = filtered_config[triple]
                next_value = agg([old_value, value])
                logging.warn(
                    "%s %s for item %s refers to the same drop table as an "
                    "existing config, where the value was %s, and is set to %s "
                    "now. Using %s ...",
                    map_name[:-1],
                    mob_event_id,
                    item_name,
                    old_value,
                    value,
                    next_value,
                )
            else:
                next_value = value

            filtered_config[triple] = next_value
            unspecified_triplets.remove(triple)

    if OTHER_STANDARD_ID in freq_per_mob_event:
        value = 1.0 / freq_per_mob_event[OTHER_STANDARD_ID]
        for triple in unspecified_triplets:
            filtered_config[triple] = value

    for triple, prob in filtered_config.items():
        ckb.crate_group_register[triple].inject(
            ir_id=ir_id,
            prob=prob,
            ignore_any_crate_prob=ignore_any_crate_prob,
            agg=agg,
        )


def inject_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    freq_per_crate_type: Dict[str, float],
    agg: Callable = min,
    extra_id_str: str = "",
    allowed_triple: Optional[Tuple[int, int, int]] = None,
    other_crate_types: Optional[Set[str]] = None,
) -> None:
    crate_target_probs = {}

    for crate_type, freq in freq_per_crate_type.items():
        if crate_type == OTHER_STANDARD_KEYWORD:
            continue

        crate_index = ckb.knowledge_base.crate_name_order_map[crate_type]
        value = 1.0 / freq

        if crate_index in crate_target_probs:
            old_value = crate_target_probs[crate_index]
            next_value = agg([old_value, value])
            logging.warn(
                "%sCRATE type %s for item %s refers to the same drop table as an "
                "existing config, where the value was %s, and is set to %s "
                "now. Using %s ...",
                extra_id_str,
                crate_type,
                item_name,
                old_value,
                value,
                next_value,
            )
        else:
            next_value = value

        crate_target_probs[crate_index] = next_value

    if OTHER_STANDARD_KEYWORD in freq_per_crate_type:
        value = 1.0 / freq_per_crate_type[OTHER_STANDARD_KEYWORD]

        other_crate_types = other_crate_types or {
            # do not add ETC crates by default
            "Standard",
            "Special",
            "Sooper",
            "Sooper Dooper",
        }

        for crate_type in other_crate_types:
            crate_index = ckb.knowledge_base.crate_name_order_map[crate_type]
            crate_target_probs.setdefault(crate_index, value)

    found_triples = (
        ckb.ir_triples[ir_id].copy() if allowed_triple is None else {allowed_triple}
    )

    for triple in found_triples:
        # check if indexes valid here, can't do it sooner for now, no cdt to talk about
        cgn_probs = ckb.crate_group_register[triple].generate_splits(
            ir_id=ir_id,
            crate_target_probs=crate_target_probs,
        )

        cgr_update_dict = {
            (cgn.cdc_id, cgn.cdt_id, cgn.is_id): cgn for cgn, _, _ in cgn_probs
        }
        ckb.crate_group_register.update(cgr_update_dict)
        ckb.ir_triples[ir_id].update(cgr_update_dict.keys())

        for cgn, main_index, prob in cgn_probs:
            cgn.crate_nodes[main_index].inject(ir_id=ir_id, prob=prob, agg=agg)


def inject_mob_and_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    freq_per_mob_and_crate_type: Dict[int, Dict[str, float]],
    agg: Callable = min,
) -> None:
    unspecified_triplets = ckb.ir_triples[ir_id].copy()

    for mob_id, freq_per_crate_type in freq_per_mob_and_crate_type.items():
        if mob_id == OTHER_STANDARD_ID:
            continue

        mob = ckb.knowledge_base.drops["Mobs"][mob_id]
        mob_drop = ckb.knowledge_base.drops["MobDrops"][mob["MobDropID"]]

        cdt_id = mob_drop["CrateDropTypeID"]
        cdt = ckb.knowledge_base.drops["CrateDropTypes"][cdt_id]
        found_triples = set()

        for crate_id in cdt["CrateIDs"]:
            crate = ckb.knowledge_base.drops["Crates"][crate_id]
            triple = (mob_drop["CrateDropChanceID"], cdt_id, crate["ItemSetID"])

            if triple in ckb.ir_triples[ir_id]:
                found_triples.add(triple)

        for triple in found_triples:
            inject_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_name,
                freq_per_crate_type=freq_per_crate_type,
                agg=agg,
                extra_id_str=f"Mob {mob_id} ",
                allowed_triple=triple,
            )

            unspecified_triplets.remove(triple)

    if OTHER_STANDARD_ID in freq_per_mob_and_crate_type:
        for triple in unspecified_triplets:
            inject_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_name,
                freq_per_crate_type=freq_per_mob_and_crate_type[OTHER_STANDARD_ID],
                agg=agg,
                extra_id_str=f"Crate group {triple} ",
                allowed_triple=triple,
            )


def inject_racing_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_type: Dict[str, float],
    epid: int,
    agg: Callable = min,
) -> None:
    racing = ckb.knowledge_base.drops["Racing"][epid]
    allowed_crate_types = {
        ckb.knowledge_base.crate_order_name_map[i]: crate_id
        for i, crate_id in enumerate(reversed(racing["Rewards"]))
        if crate_id > 0
    }
    unused_crate_types = set(allowed_crate_types.keys())

    for crate_type, freq in freq_per_crate_type.items():
        if crate_type == OTHER_STANDARD_KEYWORD:
            continue

        if crate_type not in allowed_crate_types:
            logging.warn(
                "IZ %s %s CRATE type %s is not a valid reward, skipping ...",
                epid,
                racing["EPNAme"],
                crate_type,
            )
            continue

        crate_id = allowed_crate_types[crate_type]
        prob = 1.0 / freq
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)
        unused_crate_types.remove(crate_type)

    if OTHER_STANDARD_KEYWORD in freq_per_crate_type:
        prob = 1.0 / freq_per_crate_type[OTHER_STANDARD_KEYWORD]

        for crate_type in unused_crate_types:
            crate_id = allowed_crate_types[crate_type]
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)


def inject_racing_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_iz_and_crate: Dict[int, Dict[str, float]],
    agg: Callable = min,
) -> None:
    unused_izs = set(ckb.knowledge_base.iz_name_id_map.values())

    for epid, freq_per_crate_type in freq_per_iz_and_crate.items():
        if epid == OTHER_STANDARD_ID:
            continue

        inject_racing_crate_type_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_crate_type=freq_per_crate_type,
            epid=epid,
            agg=agg,
        )
        unused_izs.remove(epid)

    if OTHER_STANDARD_ID in freq_per_iz_and_crate:
        freq_per_crate_type = freq_per_iz_and_crate[OTHER_STANDARD_ID]

        for epid in unused_izs:
            inject_racing_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_crate_type=freq_per_crate_type,
                epid=epid,
                agg=agg,
            )


def inject_mission_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_mission_level: Dict[int, float],
    agg: Callable = min,
) -> None:
    unused_levels = set(ckb.knowledge_base.mission_level_crate_id_map.keys())

    for level, freq in freq_per_mission_level.items():
        prob = 1.0 / freq

        for crate_id in ckb.knowledge_base.mission_level_crate_id_map[level]:
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)

        unused_levels.remove(level)

    if OTHER_STANDARD_ID in freq_per_mission_level:
        prob = 1.0 / freq_per_mission_level[OTHER_STANDARD_ID]

        for level in unused_levels:
            for crate_id in ckb.knowledge_base.mission_level_crate_id_map[level]:
                ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)


def inject_egger_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_type: Dict[str, float],
    zone: str,
    agg: Callable = min,
) -> None:
    mystery_egg_crates = ckb.knowledge_base.zone_to_mystery_egger_map[zone]
    unused_crate_types = set(mystery_egg_crates.keys())

    for crate_type, freq in freq_per_crate_type.items():
        if crate_type == OTHER_STANDARD_KEYWORD:
            continue

        if crate_type not in mystery_egg_crates:
            logging.warn(
                "Zone %s EGGER has no CRATE type %s, skipping ...",
                zone,
                crate_type,
            )
            continue

        crate_id = mystery_egg_crates[crate_type]
        prob = 1.0 / freq
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)
        unused_crate_types.remove(crate_type)

    if OTHER_STANDARD_KEYWORD in freq_per_crate_type:
        prob = 1.0 / freq_per_crate_type[OTHER_STANDARD_KEYWORD]

        for crate_type in unused_crate_types:
            crate_id = mystery_egg_crates[crate_id]
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)


def inject_egger_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_zone_and_crate_type: Dict[str, Dict[str, float]],
    agg: Callable = min,
) -> None:
    unused_zones = set(ckb.knowledge_base.zone_to_egger_map.keys())

    for zone, freq_per_crate_type in freq_per_zone_and_crate_type.items():
        if zone == OTHER_STANDARD_KEYWORD:
            continue

        inject_egger_crate_type_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_crate_type=freq_per_crate_type,
            zone=zone,
            agg=agg,
        )
        unused_zones.remove(zone)

    if OTHER_STANDARD_KEYWORD in freq_per_zone_and_crate_type:
        freq_per_crate_type = freq_per_zone_and_crate_type[OTHER_STANDARD_KEYWORD]

        for zone in unused_zones:
            inject_egger_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_crate_type=freq_per_crate_type,
                zone=zone,
                agg=agg,
            )


def inject_golden_egg_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_area: Dict[str, float],
    agg: Callable = min,
) -> None:
    filtered_config = {}
    unused_areas = set(ckb.knowledge_base.area_eggs.keys())

    for area, freq in freq_per_area.items():
        if area == OTHER_STANDARD_KEYWORD:
            continue

        value = 1.0 / freq

        for crate_id in ckb.knowledge_base.area_eggs[area]:
            if crate_id in filtered_config:
                old_value = filtered_config[crate_id]
                next_value = agg([old_value, value])
                logging.warn(
                    "Area %s for Golden Egg refers to the same CRATE ID %s with an "
                    "existing config, where the value was %s, and is set to %s "
                    "now. Using %s ...",
                    area,
                    crate_id,
                    old_value,
                    value,
                    next_value,
                )
            else:
                next_value = value

            filtered_config[crate_id] = next_value

        unused_areas.remove(area)

    if OTHER_STANDARD_KEYWORD in freq_per_area:
        value = 1.0 / freq_per_area[OTHER_STANDARD_KEYWORD]
        for area in unused_areas:
            filtered_config[crate_id] = value

    for crate_id, prob in filtered_config.items():
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)


def inject_named_crate_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_id: Dict[int, float],
    agg: Callable = min,
) -> None:
    for crate_id, freq in freq_per_crate_id.items():
        prob = 1.0 / freq
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob, agg=agg)


def alter_chances(
    knowledge_base: KnowledgeBase,
    item_configs: Optional[List[ItemConfig]],
    agg: Callable = min,
) -> None:
    if not item_configs:
        logging.warn("No valid item configs found, skipping the calculation step...")
        return

    ckb = ConfigKnowledgeBase(knowledge_base, item_configs, agg=agg)

    for item_config in item_configs:
        ir_id = ckb.get_ir_id(item_config)

        if item_config.kills_till_item is not None:
            inject_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                freq=item_config.kills_till_item,
                ignore_any_crate_prob=False,
                agg=agg,
            )

        if item_config.kills_till_item_per_mob:
            inject_mob_event_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                map_name="Mobs",
                freq_per_mob_event=item_config.kills_till_item_per_mob,
                ignore_any_crate_prob=False,
                agg=agg,
            )

        if item_config.crates_till_item is not None:
            inject_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                freq=item_config.crates_till_item,
                ignore_any_crate_prob=True,
                agg=agg,
            )

        if item_config.crates_till_item_per_mob:
            inject_mob_event_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                map_name="Mobs",
                freq_per_mob_event=item_config.crates_till_item_per_mob,
                ignore_any_crate_prob=True,
                agg=agg,
            )

        if item_config.crates_till_item_per_crate_type:
            inject_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                freq_per_crate_type=item_config.crates_till_item_per_crate_type,
                agg=agg,
            )

        crates_till_item_per_crate_type_and_mob = (
            item_config.crates_till_item_per_crate_type_and_mob or {}
        )
        crates_till_item_per_mob_and_crate_type = (
            item_config.crates_till_item_per_mob_and_crate_type or {}
        )

        for (
            crate_type,
            crates_till_item_per_mob,
        ) in crates_till_item_per_crate_type_and_mob.items():
            for mob_id, num_crates in crates_till_item_per_mob.items():
                if mob_id not in crates_till_item_per_mob_and_crate_type:
                    crates_till_item_per_mob_and_crate_type[mob_id] = {}

                crates_till_item_per_mob_and_crate_type[mob_id][crate_type] = num_crates

        if crates_till_item_per_mob_and_crate_type:
            inject_mob_and_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                freq_per_mob_and_crate_type=crates_till_item_per_mob_and_crate_type,
                agg=agg,
            )

        if item_config.event_kills_till_item:
            inject_mob_event_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                map_name="Events",
                freq_per_mob_event=item_config.event_kills_till_item,
                ignore_any_crate_prob=False,
                agg=agg,
            )

        if item_config.event_crates_till_item:
            inject_mob_event_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_config.name,
                map_name="Events",
                freq_per_mob_event=item_config.event_crates_till_item,
                ignore_any_crate_prob=True,
                agg=agg,
            )

        if item_config.race_crates_till_item:
            inject_racing_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_iz_and_crate=item_config.race_crates_till_item,
                agg=agg,
            )

        if item_config.mission_crates_till_item:
            inject_mission_freq_values(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_mission_level=item_config.mission_crates_till_item,
                agg=agg,
            )

        if item_config.egger_eggs_till_item:
            inject_egger_freq_values(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_zone_and_crate_type=item_config.egger_eggs_till_item,
                agg=agg,
            )

        if item_config.golden_eggs_till_item:
            inject_golden_egg_freq_values(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_area=item_config.golden_eggs_till_item,
                agg=agg,
            )

        if item_config.named_crates_till_item:
            inject_named_crate_freq_values(
                ckb=ckb,
                ir_id=ir_id,
                freq_per_crate_id=item_config.named_crates_till_item,
                agg=agg,
            )

    for itemset_node in ckb.itemset_register.values():
        itemset_node.alter_drops_values(agg=agg)

    log_output_freqs(knowledge_base, item_configs, agg=agg)


def log_output_freqs(
    knowledge_base: KnowledgeBase,
    item_configs: Optional[List[ItemConfig]],
    agg: Callable = min,
) -> None:
    pass

    if not item_configs:
        logging.warn("No valid item configs found, skipping the calculation step...")
        return

    ckb = ConfigKnowledgeBase(knowledge_base, item_configs, agg=agg)

    all_crates = defaultdict(set)
    all_triples = defaultdict(set)

    for c_id_set in ckb.ir_c_ids.values():
        for c_id in c_id_set:
            is_id = knowledge_base.drops["Crates"][c_id]["ItemSetID"]
            all_crates[is_id].add(c_id)

    for triple_set in ckb.ir_triples.values():
        for cdc_id, cdt_id, _ in triple_set:
            for crate_id in knowledge_base.drops["CrateDropTypes"][cdt_id]["CrateIDs"]:
                is_id = knowledge_base.drops["Crates"][crate_id]["ItemSetID"]
                all_triples[(cdc_id, cdt_id)].add(is_id)

    for (cdc_id, cdt_id), is_ids in all_triples.items():
        all_prob_info = {
            "Boys": {},
            "Girls": {},
        }
        desc_text = ""

        for is_id in is_ids:
            tpl = (cdc_id, cdt_id)
            triple = (*tpl, is_id)
            mob_ids = ckb.crate_drop_m_ids.get(tpl, set())
            event_ids = ckb.crate_drop_e_ids.get(tpl, set())

            if not mob_ids and not event_ids:
                continue

            mob_text = ", ".join(
                f"mob {ckb.knowledge_base.npc_map[mob_id]['m_strName']} ({mob_id})"
                for mob_id in mob_ids
            )
            event_text = ", ".join(
                f"event {ckb.knowledge_base.event_id_name_map[event_id]} ({event_id})"
                for event_id in event_ids
            )
            desc_text = " and ".join(s for s in [mob_text, event_text] if s)

            for i, gender_name in enumerate(all_prob_info.keys()):
                if triple not in ckb.crate_group_register:
                    ckb.crate_group_register[triple] = CrateGroupNode(
                        knowledge_base=knowledge_base,
                        itemset_register=ckb.itemset_register,
                        crate_register=ckb.crate_register,
                        cdc_id=cdc_id,
                        cdt_id=cdt_id,
                        is_id=is_id,
                        agg=agg,
                    )

                cgn = ckb.crate_group_register[triple]
                any_crate_prob = cgn.cdc["DropChance"] / cgn.cdc["DropChanceTotal"]

                for index, c_id in enumerate(cgn.cdt["CrateIDs"]):
                    is_id = knowledge_base.drops["Crates"][c_id]["ItemSetID"]
                    if index in cgn.crate_nodes and is_id in all_crates:
                        del all_crates[is_id]

                for ir_id, prob in cgn.drop_pool(gender_id=(i + 1)).items():
                    item_tuple = ckb.knowledge_base.irid_tuple_map[ir_id]
                    item_info = ckb.knowledge_base.item_map[item_tuple]
                    item_str = f"{item_info['m_strName'].strip()} {item_tuple}"

                    current_prob = (
                        prob
                        + all_prob_info[gender_name].get(
                            item_str, {"Probability": 0.0}
                        )["Probability"]
                    )
                    freq = inv(current_prob)
                    freq_crates = freq * any_crate_prob

                    all_prob_info[gender_name][item_str] = {
                        "Probability": current_prob,
                        "Kills to Item": round(freq, 2),
                        "Crates to Item": round(freq_crates, 2),
                    }

        logging.info(
            "Final values for %s: %s\nBS: %s GS: %s",
            desc_text,
            json.dumps(all_prob_info, indent=4),
            sum(prob["Probability"] for prob in all_prob_info["Boys"].values()),
            sum(prob["Probability"] for prob in all_prob_info["Girls"].values()),
        )

    for is_id, c_ids in all_crates.items():
        all_prob_info = {
            "Boys": {},
            "Girls": {},
        }
        desc_text = ""

        for c_id in c_ids:
            crate_node = ckb.crate_register[c_id]
            crate_info = ckb.knowledge_base.item_map[(9, c_id)]
            desc_text = "{} {} (CRATE {})".format(
                crate_info["m_strName"].strip(),
                crate_info["m_strComment"].strip()[:20],
                ', '.join(map(str, sorted(c_ids))),
            )

            for gender_id, gender_name in {1: "Boys", 2: "Girls"}.items():
                all_prob_info[gender_name] = {}
                for ir_id, prob in crate_node.drop_pool(gender_id=gender_id).items():
                    item_tuple = ckb.knowledge_base.irid_tuple_map[ir_id]
                    item_info = ckb.knowledge_base.item_map[item_tuple]
                    item_str = f"{item_info['m_strName'].strip()} {item_tuple}"

                    current_prob = (
                        prob
                        + all_prob_info[gender_name].get(
                            item_str, {"Probability": 0.0}
                        )["Probability"]
                    )
                    freq = inv(current_prob)

                    all_prob_info[gender_name][item_str] = {
                        "Probability": current_prob,
                        "Number of this type of CRATE to Item": round(freq, 2),
                    }

        logging.info(
            "Final values for %s: %s\nBS: %s GS: %s",
            desc_text,
            json.dumps(all_prob_info, indent=4),
            sum(prob["Probability"] for prob in all_prob_info["Boys"].values()),
            sum(prob["Probability"] for prob in all_prob_info["Girls"].values()),
        )
