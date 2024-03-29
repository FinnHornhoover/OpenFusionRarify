import json
import logging
from functools import wraps
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from rarify.drops import Data
from rarify.knowledge_base import KnowledgeBase
from rarify.config import ItemConfig, OTHER_STANDARD_ID, OTHER_STANDARD_KEYWORD
from rarify.math import inv, rel_diff, normalize, INF


class ItemSetNode:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        is_id: int,
        agg: Callable,
        rel_tol: float,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.is_id = is_id
        self.agg = agg
        self.rel_tol = rel_tol

        self.itemset: Data = knowledge_base.drops["ItemSets"][is_id]
        self.probs_to_change = {
            gender_id: {
                rarity_id: {} for rarity_id in knowledge_base.rarity_map.values()
            }
            for gender_id in knowledge_base.gender_map.values()
        }

    def effective_rarity_id(self, ir_id: int) -> int:
        if self.itemset["IgnoreRarity"]:
            return 0
        return self.itemset["AlterRarityMap"].get(
            str(ir_id),
            self.knowledge_base.item_map[
                self.knowledge_base.ir_id_to_tuple_map[ir_id]
            ].get("m_iRarity", 0),
        )

    def effective_gender_id(self, ir_id: int) -> int:
        if self.itemset["IgnoreGender"]:
            return 0
        return self.itemset["AlterGenderMap"].get(
            str(ir_id),
            self.knowledge_base.item_map[
                self.knowledge_base.ir_id_to_tuple_map[ir_id]
            ].get("m_iReqSex", 0),
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
        weights_scaled = normalize(weights_filtered)

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
                else (1.0 - total_target_weight)
                * weight
                * inv(total_non_target_weight, inf=0.0)
            )
            for ir_id, weight in weights_scaled.items()
        }

    def inject(self, ir_id: int, prob: float) -> None:
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

                    pool_dict[ir_id] = self.agg([pool_dict.get(ir_id, prob), prob])

    def preview_inject(self, ir_id: int, prob: float) -> bool:
        item_gender_id = self.effective_gender_id(ir_id)
        item_rarity_id = self.effective_rarity_id(ir_id)

        success = True
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
                    success = (
                        success
                        and other_probs_sum + prob <= 1.0
                        and ir_id in self.itemset["ItemReferenceIDs"]
                    )

        return success


class CrateNode:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        itemset_register: Dict[int, ItemSetNode],
        c_id: int,
        agg: Callable,
        rel_tol: float,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.itemset_register = itemset_register
        self.c_id = c_id
        self.agg = agg
        self.rel_tol = rel_tol

        self.crate = knowledge_base.drops["Crates"][c_id]
        self.rarity_weights = knowledge_base.drops["RarityWeights"][
            self.crate["RarityWeightID"]
        ]

        is_id = self.crate["ItemSetID"]
        if is_id not in self.itemset_register:
            self.itemset_register[is_id] = ItemSetNode(
                knowledge_base=knowledge_base,
                is_id=is_id,
                agg=agg,
                rel_tol=self.rel_tol,
            )

        self.itemset_node = itemset_register[is_id]

    def allowed_rarities(self, gender_id: int) -> Dict[int, int]:
        return {
            rarity_id: weight
            for rarity_id, weight in zip(
                self.knowledge_base.rarity_map.values(),
                self.rarity_weights["Weights"],
            )
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
        item_tuple = self.knowledge_base.ir_id_to_tuple_map[ir_id]
        item_gender_id = self.knowledge_base.item_map[item_tuple].get("m_iReqSex", 1)
        item_rarity_id = self.knowledge_base.item_map[item_tuple].get("m_iRarity", 1)

        rarities = self.allowed_rarities(item_gender_id)
        lower_rarities = [
            rarity_id for rarity_id in rarities if rarity_id <= item_rarity_id
        ]
        upper_rarities = [
            rarity_id for rarity_id in rarities if rarity_id > item_rarity_id
        ]

        if lower_rarities:
            changed_rarity = max(lower_rarities)
        elif upper_rarities:
            changed_rarity = min(upper_rarities)
        else:
            logging.warn(
                "Could not inject item %s no viable rarity class found in CRATE %s",
                ir_id,
                self.c_id,
            )
            return

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
                discounted_prob = INF
                logging.debug(
                    "Item %s probability %s is inf due to rarity_id %s and weights %s from crate %s to item set %s",
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
        for ir_id, prob in (target_weights or {}).items():
            if prob > 0.0:
                self.register(ir_id)

        rarity_agg = defaultdict(float)

        rw_weights = self.rarity_weights["Weights"]
        rw_sum = sum(self.allowed_rarities(gender_id).values())

        for rarity_id, rw in zip(self.knowledge_base.rarity_map.values(), rw_weights):
            pool = self.itemset_node.drop_pool(
                rarity_id=rarity_id,
                gender_id=gender_id,
                target_weights={
                    ir_id: self.discount_explained_prob(ir_id, weight)
                    for ir_id, weight in (target_weights or {}).items()
                },
            )
            for ir_id, weight in pool.items():
                rarity_agg[ir_id] += weight * rw * inv(rw_sum, inf=0.0)

        return rarity_agg

    def inject(self, ir_id: int, prob: float) -> None:
        if self.preview_inject(ir_id=ir_id, prob=prob):
            self.itemset_node.inject(
                ir_id=ir_id,
                prob=self.discount_explained_prob(ir_id, prob),
            )

    def preview_inject(self, ir_id: int, prob: float) -> bool:
        if prob > 0.0:
            self.register(ir_id)
        return self.itemset_node.preview_inject(
            ir_id=ir_id,
            prob=self.discount_explained_prob(ir_id, prob),
        )


class CrateGroupNode:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        itemset_register: Dict[int, ItemSetNode],
        crate_register: Dict[int, CrateNode],
        ir_is_ids: Dict[int, Set[int]],
        cdc_id: int,
        cdt_id: int,
        agg: Callable,
        rel_tol: float,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.itemset_register = itemset_register
        self.crate_register = crate_register
        self.ir_is_ids = ir_is_ids
        self.cdc_id = cdc_id
        self.cdt_id = cdt_id
        self.agg = agg
        self.rel_tol = rel_tol

        self.cdc = knowledge_base.drops["CrateDropChances"][cdc_id]
        self.cdt = knowledge_base.drops["CrateDropTypes"][cdt_id]

        self.refresh_crates()

    def refresh_crates(self) -> None:
        self.ir_is_ids.update(
            {
                ir_id: {
                    is_id
                    for map_name_is, is_id in self.knowledge_base.drops.references.get(
                        ("ItemReferences", ir_id), []
                    )
                    if map_name_is == "ItemSets"
                }
                for ir_id in self.ir_is_ids
            }
        )

        self.is_id_index: Dict[int, Set[int]] = defaultdict(set)
        self.crate_nodes: List[CrateNode] = []

        for index, crate_id in enumerate(self.cdt["CrateIDs"]):
            if crate_id not in self.crate_register:
                self.crate_register[crate_id] = CrateNode(
                    knowledge_base=self.knowledge_base,
                    itemset_register=self.itemset_register,
                    c_id=crate_id,
                    agg=self.agg,
                    rel_tol=self.rel_tol,
                )
            crate_node = self.crate_register[crate_id]
            self.is_id_index[crate_node.itemset_node.is_id].add(index)
            self.crate_nodes.append(crate_node)

    def register(self, ir_id: int, crate_types: Optional[Set[int]] = None) -> None:
        if not crate_types and any(
            ir_id in self.drop_pool(gender_id=gender_id)
            for gender_id in self.knowledge_base.gender_map.values()
        ):
            return

        # do not add ETC crates by default
        crate_types = {
            min(crate_index, len(self.crate_nodes) - 1)
            for crate_index in (crate_types or {1, 2, 3, 4})
        }

        for crate_index in crate_types:
            crate_node = self.crate_nodes[crate_index]

            crate_gender_id = crate_node.itemset_node.effective_gender_id(ir_id)
            crate_rarity_id = crate_node.itemset_node.effective_rarity_id(ir_id)
            # only inject if the exact rarity is available here
            if crate_rarity_id in crate_node.allowed_rarities(crate_gender_id):
                crate_node.register(ir_id=ir_id)

            if ir_id not in self.ir_is_ids:
                self.ir_is_ids = set()
            self.ir_is_ids[ir_id].add(crate_node.itemset_node.is_id)

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
            discounted_prob = INF
        else:
            discounted_prob = (
                prob
                * (cdwt / cdw[crate_index])
                * inv(1.0 if ignore_any_crate_prob else any_crate_prob)
            )

        logging.debug(
            "Probability %s discounted to %s from crate group %s to crate %s",
            prob,
            discounted_prob,
            (self.cdc_id, self.cdt_id),
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

        for index, crate_node in enumerate(self.crate_nodes):
            pool = crate_node.drop_pool(
                gender_id=gender_id,
                target_weights={
                    ir_id: self.discount_explained_prob(index, weight)
                    for ir_id, weight in (target_weights or {}).items()
                },
            )
            for ir_id, weight in pool.items():
                group_agg[ir_id] += any_crate_prob * cdw[index] * weight / cdwt

        return group_agg

    def generate_split_injections(
        self,
        ir_id: int,
        crate_target_probs: Dict[int, float],
    ) -> Dict[int, float]:
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

        groups = {}
        for current_is_id, indices in self.is_id_index.items():
            if current_is_id not in self.ir_is_ids[ir_id]:
                continue

            freq_groups = defaultdict(list)

            def groups_append(item_freq: float, index: int) -> None:
                added = False
                for freq, freq_group in freq_groups.items():
                    if rel_diff(item_freq, freq) < self.rel_tol:
                        freq_group.append(index)
                        added = True
                        break

                if not added:
                    freq_groups[item_freq].append(index)

            # stable sort strats
            left_out_indices = []
            for index in indices:
                if index not in fixed_crate_target_probs:
                    continue

                crate_node = self.crate_nodes[index]
                # this may not work for rarity_id=0
                prob_list = []
                for gender_id in self.knowledge_base.gender_map.values():
                    weights = crate_node.drop_pool(gender_id=gender_id)
                    if ir_id in weights:
                        prob_list.append(
                            crate_node.discount_explained_prob(ir_id, weights[ir_id])
                        )

                # crate doesn't or cannot contain item
                if not prob_list:
                    left_out_indices.append(index)
                    continue

                item_freq = inv(self.agg(prob_list))

                # if later crates of the same itemset can contain the item
                # group them with this one
                for loi in left_out_indices:
                    groups_append(item_freq, loi)
                left_out_indices.clear()

                groups_append(item_freq, index)

            # if all crates were skipped, it means we cannot drop this item
            for loi in left_out_indices:
                groups_append(INF, loi)

            # now group up the changes requested
            for index in indices:
                if index not in fixed_crate_target_probs:
                    continue

                desired_prob = self.crate_nodes[index].discount_explained_prob(
                    ir_id=ir_id,
                    prob=fixed_crate_target_probs[index],
                )
                desired_freq = inv(desired_prob)

                for freq_group in freq_groups.values():
                    if index in freq_group:
                        freq_group.remove(index)

                if self.crate_nodes[index].itemset_node.preview_inject(
                    ir_id=ir_id, prob=desired_prob
                ):
                    groups_append(desired_freq, index)

            # eliminate empty groups
            freq_groups = {
                freq: freq_group
                for freq, freq_group in freq_groups.items()
                if freq_group
            }

            # sort the largest groups to the top
            # in the case of a tie, this should still leave the original group at the top
            sorted_freq_groups = sorted(
                freq_groups.items(),
                key=lambda t: len(t[1]),
                reverse=True,
            )

            # index and is_id management
            for i, (freq, freq_group) in enumerate(sorted_freq_groups):
                if i == 0:
                    is_id = current_is_id
                else:
                    new_itemset = self.itemset_register[
                        current_is_id
                    ].itemset.deepcopy()
                    is_id = self.knowledge_base.drops["ItemSets"].add(new_itemset)

                    for _ir_id in new_itemset["ItemReferenceIDs"]:
                        self.knowledge_base.drops.references[
                            ("ItemReferences", _ir_id)
                        ].add(("ItemSets", is_id))

                main_index = None
                crate_prob = inv(freq)

                for index in freq_group:
                    crate_id = self.cdt["CrateIDs"][index]
                    crate = self.knowledge_base.drops["Crates"][crate_id]
                    # this should also fix the crate references
                    crate["ItemSetID"] = is_id

                    # force reload
                    self.crate_register[crate_id] = CrateNode(
                        knowledge_base=self.knowledge_base,
                        itemset_register=self.itemset_register,
                        c_id=crate_id,
                        agg=self.agg,
                        rel_tol=self.rel_tol,
                    )

                    if main_index is None:
                        if self.crate_register[crate_id].itemset_node.preview_inject(
                            ir_id=ir_id, prob=crate_prob
                        ):
                            main_index = index
                        else:
                            logging.warn(
                                "Rejecting injection %s: %s for CRATE at index %s (ID %s) ...",
                                ir_id,
                                crate_prob,
                                index,
                                self.cdt["CrateIDs"][index],
                            )

                if main_index is None:
                    logging.warn(
                        "Rejecting all injections %s: %s for CRATE at indices %s (IDs %s) ...",
                        ir_id,
                        crate_prob,
                        freq_group,
                        [self.cdt["CrateIDs"][index] for index in freq_group],
                    )
                else:
                    groups[main_index] = crate_prob

        # refresh the state
        self.refresh_crates()

        return groups

    def generate_whole_injections(
        self,
        ir_id: int,
        prob: float,
        ignore_any_crate_prob: bool = False,
    ) -> Dict[int, float]:
        cdw = self.cdc["CrateTypeDropWeights"]
        cdwt = sum(cdw)

        crate_injections: Dict[int, float] = {}
        main_probs: Dict[int, float] = {}
        sum_probs = 0.0

        for is_id, indices in self.is_id_index.items():
            if is_id not in self.ir_is_ids[ir_id]:
                continue

            main_index = None
            main_crate_prob = None

            for index in indices:
                crate_prob = self.discount_explained_prob(
                    crate_index=index,
                    prob=prob,
                    ignore_any_crate_prob=ignore_any_crate_prob,
                )

                if self.crate_nodes[index].preview_inject(ir_id=ir_id, prob=crate_prob):
                    main_index = index
                    main_crate_prob = crate_prob
                    break

            if main_index is None:
                continue

            main_crate_node = self.crate_nodes[main_index]
            gender_id = max(1, main_crate_node.itemset_node.effective_gender_id(ir_id))
            rarity_id = main_crate_node.itemset_node.effective_rarity_id(ir_id)

            is_prob = main_crate_node.discount_explained_prob(
                ir_id=ir_id, prob=main_crate_prob
            )
            main_prob = cdw[main_index] * main_crate_prob / cdwt
            main_probs[main_index] = main_prob
            crate_injections[main_index] = is_prob

            for index in indices:
                crate_node = self.crate_nodes[index]
                rw_weights = crate_node.allowed_rarities(gender_id)
                rw_sum = sum(rw_weights.values())

                crate_prob = sum(
                    [
                        is_prob * rw / rw_sum
                        for r_id, rw in rw_weights.items()
                        if rarity_id in [0, r_id]
                    ]
                )
                sum_probs += cdw[index] * crate_prob / cdwt

        return {
            index: main_probs[index] * is_prob * inv(sum_probs, inf=0.0)
            for index, is_prob in crate_injections.items()
        }

    def inject(
        self,
        ir_id: int,
        prob: Union[float, Dict[int, float]],
        ignore_any_crate_prob: bool = False,
    ) -> None:
        if isinstance(prob, dict):
            self.register(
                ir_id=ir_id, crate_types={i for i, p in prob.items() if p > 0.0}
            )
            crate_injections = self.generate_split_injections(
                ir_id=ir_id, crate_target_probs=prob
            )
        else:
            if prob > 0.0:
                self.register(ir_id=ir_id)
            crate_injections = self.generate_whole_injections(
                ir_id=ir_id, prob=prob, ignore_any_crate_prob=ignore_any_crate_prob
            )

        for crate_index, is_prob in crate_injections.items():
            self.crate_nodes[crate_index].itemset_node.inject(ir_id=ir_id, prob=is_prob)


class ConfigKnowledgeBase:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        item_configs: List[ItemConfig],
        agg: Callable,
        rel_tol: float,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.agg = agg
        self.rel_tol = rel_tol

        references = knowledge_base.drops.references
        ir_ids = [self.get_ir_id(item_config) for item_config in item_configs] or list(
            knowledge_base.drops["ItemReferences"].ikeys()
        )
        self.ir_is_ids = {
            ir_id: {
                is_id
                for map_name_is, is_id in references.get(("ItemReferences", ir_id), [])
                if map_name_is == "ItemSets"
            }
            for ir_id in ir_ids
        }
        self.itemset_register = {
            # the filtering here won't matter, but it doesn't hurt either
            is_id: ItemSetNode(
                knowledge_base=knowledge_base,
                is_id=is_id,
                agg=agg,
                rel_tol=rel_tol,
            )
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
            c_id: CrateNode(
                knowledge_base=knowledge_base,
                itemset_register=self.itemset_register,
                c_id=c_id,
                agg=agg,
                rel_tol=rel_tol,
            )
            # pull them all, we need to recognize any and all crates
            for c_id in knowledge_base.drops["Crates"]
        }

        self.ir_tuples = defaultdict(set)
        self.crate_drop_m_ids = defaultdict(set)
        self.crate_drop_e_ids = defaultdict(set)

        for ir_id, c_ids in self.ir_c_ids.items():
            for c_id in c_ids:
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
                        self.ir_tuples[ir_id].add(tpl)

                        for map_name_me, me_id in references.get(
                            ("MobDrops", md_id), []
                        ):
                            if map_name_me == "Mobs":
                                self.crate_drop_m_ids[tpl].add(me_id)
                            elif map_name_me == "Events":
                                self.crate_drop_e_ids[tpl].add(me_id)

        self.crate_group_register = {
            (cdc_id, cdt_id): self.make_crate_group_register(cdc_id, cdt_id)
            for tpls in self.ir_tuples.values()
            for cdc_id, cdt_id in tpls
        }

    def get_ir_id(self, item_config: ItemConfig) -> int:
        item_tuple = (item_config.type, item_config.id)
        ir_id = self.knowledge_base.tuple_to_ir_id_map.get(item_tuple)

        if item_tuple not in self.knowledge_base.tuple_to_ir_id_map:
            ir_id = self.knowledge_base.drops["ItemReferences"].add(
                {
                    "ItemReferenceID": -1,
                    "ItemID": item_config.id,
                    "Type": item_config.type,
                }
            )
            self.knowledge_base.tuple_to_ir_id_map[item_tuple] = ir_id
            self.knowledge_base.ir_id_to_tuple_map[ir_id] = item_tuple

        return ir_id

    def make_crate_group_register(self, cdc_id: int, cdt_id: int) -> CrateGroupNode:
        return CrateGroupNode(
            knowledge_base=self.knowledge_base,
            itemset_register=self.itemset_register,
            crate_register=self.crate_register,
            ir_is_ids=self.ir_is_ids,
            cdc_id=cdc_id,
            cdt_id=cdt_id,
            agg=self.agg,
            rel_tol=self.rel_tol,
        )


def log_entry_end(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if isinstance(v, (int, float, bool, bytes, str, list, set, dict))
        }
        logging.debug(f"Injecting start {func.__name__} Args: {args} {filtered_kwargs}")
        val = func(*args, **kwargs)
        logging.debug(f"Injecting end {func.__name__}")
        return val

    return wrapper


@log_entry_end
def inject_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq: float,
    ignore_any_crate_prob: bool = False,
) -> None:
    prob = 1.0 / freq
    for tpl in ckb.ir_tuples[ir_id]:
        ckb.crate_group_register[tpl].inject(
            ir_id=ir_id,
            prob=prob,
            ignore_any_crate_prob=ignore_any_crate_prob,
        )


@log_entry_end
def inject_mob_event_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    map_name: str,
    freq_per_mob_event: Dict[int, float],
    ignore_any_crate_prob: bool = False,
) -> None:
    filtered_config = {}
    unspecified_tuples = ckb.ir_tuples[ir_id].copy()

    for mob_event_id, freq in freq_per_mob_event.items():
        if mob_event_id == OTHER_STANDARD_ID:
            continue

        mob_event = ckb.knowledge_base.drops[map_name][mob_event_id]
        mob_drop = ckb.knowledge_base.drops["MobDrops"][mob_event["MobDropID"]]
        tpl = (mob_drop["CrateDropChanceID"], mob_drop["CrateDropTypeID"])

        if tpl not in ckb.crate_group_register:
            ckb.crate_group_register[tpl] = ckb.make_crate_group_register(*tpl)

        value = 1.0 / freq

        if tpl in filtered_config:
            old_value = filtered_config[tpl]
            next_value = ckb.agg([old_value, value])
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

        filtered_config[tpl] = next_value
        if tpl in unspecified_tuples:
            unspecified_tuples.remove(tpl)

    if OTHER_STANDARD_ID in freq_per_mob_event:
        value = 1.0 / freq_per_mob_event[OTHER_STANDARD_ID]
        for tpl in unspecified_tuples:
            filtered_config[tpl] = value

    for tpl, prob in filtered_config.items():
        ckb.crate_group_register[tpl].inject(
            ir_id=ir_id,
            prob=prob,
            ignore_any_crate_prob=ignore_any_crate_prob,
        )


@log_entry_end
def inject_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    freq_per_crate_type: Dict[str, float],
    extra_id_str: str = "",
    allowed_tuple: Optional[Tuple[int, int]] = None,
    other_crate_types: Optional[Set[str]] = None,
) -> None:
    crate_target_probs = {}

    for crate_type, freq in freq_per_crate_type.items():
        if crate_type == OTHER_STANDARD_KEYWORD:
            continue

        crate_index = ckb.knowledge_base.crate_type_to_crate_order_map[crate_type]
        value = 1.0 / freq

        if crate_index in crate_target_probs:
            old_value = crate_target_probs[crate_index]
            next_value = ckb.agg([old_value, value])
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
            crate_index = ckb.knowledge_base.crate_type_to_crate_order_map[crate_type]
            crate_target_probs.setdefault(crate_index, value)

    found_tuples = (
        ckb.ir_tuples[ir_id].copy() if allowed_tuple is None else {allowed_tuple}
    )

    for tpl in found_tuples:
        # check if indexes valid here, can't do it sooner for now, no cdt to talk about
        ckb.crate_group_register[tpl].inject(
            ir_id=ir_id,
            prob=crate_target_probs,
            ignore_any_crate_prob=True,
        )


@log_entry_end
def inject_mob_and_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    item_name: str,
    freq_per_mob_and_crate_type: Dict[int, Dict[str, float]],
) -> None:
    unspecified_tuples = ckb.ir_tuples[ir_id].copy()

    for mob_id, freq_per_crate_type in freq_per_mob_and_crate_type.items():
        if mob_id == OTHER_STANDARD_ID:
            continue

        mob = ckb.knowledge_base.drops["Mobs"][mob_id]
        mob_drop = ckb.knowledge_base.drops["MobDrops"][mob["MobDropID"]]
        tpl = (mob_drop["CrateDropChanceID"], mob_drop["CrateDropTypeID"])

        if tpl not in ckb.crate_group_register:
            ckb.crate_group_register[tpl] = ckb.make_crate_group_register(*tpl)

        inject_crate_type_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_name,
            freq_per_crate_type=freq_per_crate_type,
            extra_id_str=f"Mob {mob_id} ",
            allowed_tuple=tpl,
        )

        if tpl in unspecified_tuples:
            unspecified_tuples.remove(tpl)

    if OTHER_STANDARD_ID in freq_per_mob_and_crate_type:
        for tpl in unspecified_tuples:
            inject_crate_type_freq_value(
                ckb=ckb,
                ir_id=ir_id,
                item_name=item_name,
                freq_per_crate_type=freq_per_mob_and_crate_type[OTHER_STANDARD_ID],
                extra_id_str=f"Crate group {tpl} ",
                allowed_tuple=tpl,
            )


@log_entry_end
def inject_racing_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_type: Dict[str, float],
    epid: int,
) -> None:
    racing = ckb.knowledge_base.drops["Racing"][epid]
    allowed_crate_types = {
        ckb.knowledge_base.crate_order_to_crate_type_map[i]: crate_id
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
                racing["EPName"],
                crate_type,
            )
            continue

        crate_id = allowed_crate_types[crate_type]
        prob = 1.0 / freq
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)
        unused_crate_types.remove(crate_type)

    if OTHER_STANDARD_KEYWORD in freq_per_crate_type:
        prob = 1.0 / freq_per_crate_type[OTHER_STANDARD_KEYWORD]

        for crate_type in unused_crate_types:
            crate_id = allowed_crate_types[crate_type]
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)


@log_entry_end
def inject_racing_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_iz_and_crate: Dict[int, Dict[str, float]],
) -> None:
    unused_izs = set(ckb.knowledge_base.iz_name_to_epid_map.values())

    for epid, freq_per_crate_type in freq_per_iz_and_crate.items():
        if epid == OTHER_STANDARD_ID:
            continue

        inject_racing_crate_type_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_crate_type=freq_per_crate_type,
            epid=epid,
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
            )


@log_entry_end
def inject_mission_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_mission_level: Dict[int, float],
) -> None:
    unused_levels = set(ckb.knowledge_base.mission_level_crate_id_map.keys())

    for level, freq in freq_per_mission_level.items():
        if level == OTHER_STANDARD_ID:
            continue

        prob = 1.0 / freq

        for crate_id in ckb.knowledge_base.mission_level_crate_id_map[level]:
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)

        unused_levels.remove(level)

    if OTHER_STANDARD_ID in freq_per_mission_level:
        prob = 1.0 / freq_per_mission_level[OTHER_STANDARD_ID]

        for level in unused_levels:
            for crate_id in ckb.knowledge_base.mission_level_crate_id_map[level]:
                ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)


@log_entry_end
def inject_egger_crate_type_freq_value(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_type: Dict[str, float],
    zone: str,
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
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)
        unused_crate_types.remove(crate_type)

    if OTHER_STANDARD_KEYWORD in freq_per_crate_type:
        prob = 1.0 / freq_per_crate_type[OTHER_STANDARD_KEYWORD]

        for crate_type in unused_crate_types:
            crate_id = mystery_egg_crates[crate_id]
            ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)


@log_entry_end
def inject_egger_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_zone_and_crate_type: Dict[str, Dict[str, float]],
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
            )


@log_entry_end
def inject_golden_egg_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_area: Dict[str, float],
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
                next_value = ckb.agg([old_value, value])
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
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)


@log_entry_end
def inject_named_crate_freq_values(
    ckb: ConfigKnowledgeBase,
    ir_id: int,
    freq_per_crate_id: Dict[int, float],
) -> None:
    for crate_id, freq in freq_per_crate_id.items():
        prob = 1.0 / freq
        ckb.crate_register[crate_id].inject(ir_id=ir_id, prob=prob)


def inject_item(ckb: ConfigKnowledgeBase, item_config: ItemConfig) -> None:
    ir_id = ckb.get_ir_id(item_config)

    if item_config.kills_till_item is not None:
        inject_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq=item_config.kills_till_item,
            ignore_any_crate_prob=False,
        )

    if item_config.kills_till_item_per_mob:
        inject_mob_event_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_config.name,
            map_name="Mobs",
            freq_per_mob_event=item_config.kills_till_item_per_mob,
            ignore_any_crate_prob=False,
        )

    if item_config.crates_till_item is not None:
        inject_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq=item_config.crates_till_item,
            ignore_any_crate_prob=True,
        )

    if item_config.crates_till_item_per_mob:
        inject_mob_event_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_config.name,
            map_name="Mobs",
            freq_per_mob_event=item_config.crates_till_item_per_mob,
            ignore_any_crate_prob=True,
        )

    if item_config.crates_till_item_per_crate_type:
        inject_crate_type_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_config.name,
            freq_per_crate_type=item_config.crates_till_item_per_crate_type,
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
        )

    if item_config.event_kills_till_item:
        inject_mob_event_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_config.name,
            map_name="Events",
            freq_per_mob_event=item_config.event_kills_till_item,
            ignore_any_crate_prob=False,
        )

    if item_config.event_crates_till_item:
        inject_mob_event_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            item_name=item_config.name,
            map_name="Events",
            freq_per_mob_event=item_config.event_crates_till_item,
            ignore_any_crate_prob=True,
        )

    if item_config.race_crates_till_item:
        inject_racing_freq_value(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_iz_and_crate=item_config.race_crates_till_item,
        )

    if item_config.mission_crates_till_item:
        inject_mission_freq_values(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_mission_level=item_config.mission_crates_till_item,
        )

    if item_config.egger_eggs_till_item:
        inject_egger_freq_values(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_zone_and_crate_type=item_config.egger_eggs_till_item,
        )

    if item_config.golden_eggs_till_item:
        inject_golden_egg_freq_values(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_area=item_config.golden_eggs_till_item,
        )

    if item_config.named_crates_till_item:
        inject_named_crate_freq_values(
            ckb=ckb,
            ir_id=ir_id,
            freq_per_crate_id=item_config.named_crates_till_item,
        )


def inject_all_items(ckb: ConfigKnowledgeBase, item_configs: List[ItemConfig]) -> None:
    for item_config in item_configs:
        inject_item(ckb, item_config)


def log_output_freqs(
    prev_ckb: ConfigKnowledgeBase,
    item_configs: List[ItemConfig],
) -> None:
    ckb = ConfigKnowledgeBase(
        knowledge_base=prev_ckb.knowledge_base,
        item_configs=item_configs,
        agg=prev_ckb.agg,
        rel_tol=prev_ckb.rel_tol,
    )

    old_itemsets = prev_ckb.knowledge_base.base_drops["ItemSets"]
    new_itemsets = ckb.knowledge_base.drops["ItemSets"]

    all_itemsets = {
        is_id
        for is_id, itemset in new_itemsets.iitems()
        if len(item_configs) == 0
        or is_id not in old_itemsets
        or old_itemsets[is_id] != itemset
    }
    all_crates = defaultdict(set)
    all_tuples = set()

    for c_id_set in ckb.ir_c_ids.values():
        for c_id in c_id_set:
            is_id = ckb.knowledge_base.drops["Crates"][c_id]["ItemSetID"]
            if is_id in all_itemsets:
                all_crates[is_id].add(c_id)

    for tpls in ckb.ir_tuples.values():
        for cdc_id, cdt_id in tpls:
            for crate_id in ckb.knowledge_base.drops["CrateDropTypes"][cdt_id][
                "CrateIDs"
            ]:
                is_id = ckb.knowledge_base.drops["Crates"][crate_id]["ItemSetID"]
                if is_id in all_itemsets:
                    all_tuples.add((cdc_id, cdt_id))

    for tpl in all_tuples:
        all_prob_info = {}
        mob_ids = ckb.crate_drop_m_ids.get(tpl, set())
        event_ids = ckb.crate_drop_e_ids.get(tpl, set())

        if not mob_ids and not event_ids:
            continue

        mob_text = ", ".join(
            f"mob {ckb.knowledge_base.npc_map[mob_id]['m_strName']} ({mob_id})"
            for mob_id in sorted(mob_ids)
        )
        event_text = ", ".join(
            f"event {ckb.knowledge_base.event_id_to_event_type_map[event_id]} ({event_id})"
            for event_id in sorted(event_ids)
        )
        desc_text = " and ".join(s for s in [mob_text, event_text] if s)

        cgn = ckb.crate_group_register[tpl]
        any_crate_prob = cgn.cdc["DropChance"] / cgn.cdc["DropChanceTotal"]

        for c_id in cgn.cdt["CrateIDs"]:
            is_id = ckb.knowledge_base.drops["Crates"][c_id]["ItemSetID"]
            if is_id in all_crates:
                del all_crates[is_id]

        for gender_type, gender_id in ckb.knowledge_base.gender_map.items():
            all_prob_info[gender_type] = {}
            for ir_id, prob in cgn.drop_pool(gender_id=gender_id).items():
                item_tuple = ckb.knowledge_base.ir_id_to_tuple_map[ir_id]
                item_info = ckb.knowledge_base.item_map[item_tuple]
                item_str = f"{item_info['m_strName'].strip()} {item_tuple}"

                current_prob = (
                    prob
                    + all_prob_info[gender_type].get(item_str, {"Probability": 0.0})[
                        "Probability"
                    ]
                )
                freq = inv(current_prob)
                freq_crates = freq * any_crate_prob

                all_prob_info[gender_type][item_str] = {
                    "Probability": current_prob,
                    "Kills to Item": round(freq, 2),
                    "Crates to Item": round(freq_crates, 2),
                }

        all_prob_info = {
            gender_type: dict(
                sorted(gender_dict.items(), key=lambda t: t[1]["Probability"])
            )
            for gender_type, gender_dict in all_prob_info.items()
        }

        logging.info(
            "Final values for %s: %s\nBoy Sum: %s Girl Sum: %s",
            desc_text,
            json.dumps(all_prob_info, indent=4),
            sum(prob["Probability"] for prob in all_prob_info["Boy"].values()),
            sum(prob["Probability"] for prob in all_prob_info["Girl"].values()),
        )

    for is_id, c_ids in all_crates.items():
        all_prob_info = {}
        desc_text = ""

        for c_id in c_ids:
            crate_node = ckb.crate_register[c_id]
            crate_info = ckb.knowledge_base.item_map[(9, c_id)]
            desc_text = "{} {} (CRATE {})".format(
                crate_info["m_strName"].strip(),
                crate_info["m_strComment"].strip()[:20],
                ", ".join(map(str, sorted(c_ids))),
            )

            for gender_type, gender_id in ckb.knowledge_base.gender_map.items():
                all_prob_info[gender_type] = {}
                for ir_id, prob in crate_node.drop_pool(gender_id=gender_id).items():
                    item_tuple = ckb.knowledge_base.ir_id_to_tuple_map[ir_id]
                    item_info = ckb.knowledge_base.item_map[item_tuple]
                    item_str = f"{item_info['m_strName'].strip()} {item_tuple}"

                    current_prob = (
                        prob
                        + all_prob_info[gender_type].get(
                            item_str, {"Probability": 0.0}
                        )["Probability"]
                    )
                    freq = inv(current_prob)

                    all_prob_info[gender_type][item_str] = {
                        "Probability": current_prob,
                        "Number of this type of CRATE to Item": round(freq, 2),
                    }

        all_prob_info = {
            gender_type: dict(
                sorted(gender_dict.items(), key=lambda t: t[1]["Probability"])
            )
            for gender_type, gender_dict in all_prob_info.items()
        }

        logging.info(
            "Final values for %s: %s\nBoy Sum: %s Girl Sum: %s",
            desc_text,
            json.dumps(all_prob_info, indent=4),
            sum(prob["Probability"] for prob in all_prob_info["Boy"].values()),
            sum(prob["Probability"] for prob in all_prob_info["Girl"].values()),
        )
