import json
import logging
from operator import itemgetter
from typing import Dict, List, Set

from rarify.drops import Data
from rarify.config import Config
from rarify.knowledge_base import KnowledgeBase
from rarify.config import ItemConfig  # import this AFTER KnowledgeBase
from rarify.math import (
    inv,
    rel_diff,
    normalize,
    get_value_mode,
    ln,
    exp,
    logsumexp,
    NINF,
)
from rarify.injections import (
    ConfigKnowledgeBase,
    ItemSetNode,
    inject_all_items,
    log_output_freqs,
)


def apply_scale(prob_dict: Dict[int, float], scale: int) -> Dict[int, int]:
    return {ir_id: int(scale * value) for ir_id, value in prob_dict.items()}


def merge_pools(
    altered_ir_ids: Set[int],
    acc_dict: Dict[int, float],
    other_dict: Dict[int, float],
) -> Dict[int, float]:
    cur_acc_dict = acc_dict.copy()
    cur_other_dict = other_dict.copy()

    shared_ir_ids = [
        ir_id
        for ir_id, acc_value in cur_acc_dict.items()
        if (
            ir_id in cur_other_dict
            and acc_value > NINF
            and cur_other_dict[ir_id] > NINF
            and acc_value < 0.0
            and cur_other_dict[ir_id] < 0.0
            # do not consider altered values
            and ir_id not in altered_ir_ids
        )
    ]

    if shared_ir_ids:
        # it seems to be more accurate for both cases to take the maximum distance here, not sure why
        select_ir_id = max(
            shared_ir_ids,
            key=lambda i: abs(cur_acc_dict[i] - cur_other_dict[i]),
        )
        other_to_scale = cur_acc_dict[select_ir_id]
        acc_to_scale = cur_other_dict[select_ir_id]
        cur_other_dict = {
            ir_id: (
                # copy over values for automatically 1.0 or 0.0-ed out probabilities
                cur_acc_dict[ir_id]
                if (value == NINF or value == 0.0) and ir_id in cur_acc_dict
                else value + other_to_scale - acc_to_scale
            )
            for ir_id, value in cur_other_dict.items()
        }

    if cur_other_dict:
        log_sum = logsumexp(cur_other_dict)
        cur_acc_dict = {ir_id: value - log_sum for ir_id, value in cur_acc_dict.items()}
        cur_other_dict = {
            ir_id: value - log_sum for ir_id, value in cur_other_dict.items()
        }

    return {**cur_acc_dict, **cur_other_dict}


def search_scale(
    split_weights: Dict[int, Dict[int, Dict[int, float]]],
    rel_tol: float,
    lo: int,
    hi: int,
) -> int:
    if hi < lo:
        return -1

    mi = (lo + hi) // 2

    split_probs = {
        gender_id: {
            rarity_id: normalize(pool_dict)
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in split_weights.items()
    }
    current_weights = {
        gender_id: {
            rarity_id: apply_scale(pool_dict, mi)
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in split_weights.items()
    }
    current_probs = {
        gender_id: {
            rarity_id: normalize(pool_dict)
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in current_weights.items()
    }

    if all(
        rel_diff(
            inv(value),
            inv(split_probs[gender_id][rarity_id][ir_id]),
        )
        < rel_tol
        for gender_id, rarity_dicts in current_probs.items()
        for rarity_id, pool_dict in rarity_dicts.items()
        for ir_id, value in pool_dict.items()
    ):
        left = search_scale(split_weights, rel_tol, lo, mi - 1)
        return mi if left == -1 else left

    return search_scale(split_weights, rel_tol, mi + 1, hi)


def correct_zero_weights(int_weights: Dict[int, int]) -> Dict[int, int]:
    int_scaled_weights = int_weights.copy()

    for ir_id in int_scaled_weights:
        if int_scaled_weights[ir_id] == 0:
            max_weight_ir_id, max_weight = max(
                int_scaled_weights.items(), key=itemgetter(1)
            )
            if max_weight > 1:
                int_scaled_weights[ir_id] = 1
                int_scaled_weights[max_weight_ir_id] -= 1

    return {ir_id: weight for ir_id, weight in int_scaled_weights.items() if weight > 0}


def remove_zero_prob_entries(
    merged_probs: Dict[int, Dict[int, Dict[int, float]]]
) -> Dict[int, Dict[int, Dict[int, float]]]:
    ir_id_set = {
        ir_id
        for rarity_dicts in merged_probs.values()
        for prob_dict in rarity_dicts.values()
        for ir_id in prob_dict
    }

    zero_entries = {
        ir_id
        for ir_id in ir_id_set
        if all(
            prob_dict[ir_id] == 0.0
            for rarity_dicts in merged_probs.values()
            for prob_dict in rarity_dicts.values()
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
            for rarity_id, prob_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in merged_probs.items()
    }


def rescale_one_prob_entries(
    merged_probs: Dict[int, Dict[int, Dict[int, float]]],
    log_scaled_weights: Dict[int, float],
) -> Dict[int, float]:
    ir_id_set = {
        ir_id
        for rarity_dicts in merged_probs.values()
        for prob_dict in rarity_dicts.values()
        for ir_id in prob_dict
    }

    one_entries = {
        ir_id
        for ir_id in ir_id_set
        if all(
            len(prob_dict) == 1
            for rarity_dicts in merged_probs.values()
            for prob_dict in rarity_dicts.values()
            if ir_id in prob_dict
        )
    }

    if not one_entries:
        return log_scaled_weights

    mode = get_value_mode(log_scaled_weights)

    return {
        ir_id: mode if ir_id in one_entries else log_weight
        for ir_id, log_weight in log_scaled_weights.items()
    }


def get_scaled_int_weights(
    merged_probs: Dict[int, Dict[int, Dict[int, float]]],
    altered_ir_ids: Set[int],
    rel_tol: float,
    scale_max: int,
) -> Dict[int, int]:
    log_scaled_weights = {}

    for rarity_dicts in merged_probs.values():
        gender_scaled_weights = {}

        for pool_dict in rarity_dicts.values():
            log_pool_dict = {ir_id: ln(value) for ir_id, value in pool_dict.items()}
            gender_scaled_weights = merge_pools(
                altered_ir_ids, gender_scaled_weights, log_pool_dict
            )

        log_scaled_weights = merge_pools(
            altered_ir_ids, log_scaled_weights, gender_scaled_weights
        )

    log_scaled_weights = rescale_one_prob_entries(merged_probs, log_scaled_weights)

    log_split_weights = {
        gender_id: {
            rarity_id: {ir_id: log_scaled_weights[ir_id] for ir_id in pool_dict}
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in merged_probs.items()
    }
    split_sums = [
        logsumexp(pool_dict)
        for rarity_dicts in log_split_weights.values()
        for pool_dict in rarity_dicts.values()
        if pool_dict
    ]
    max_split_sum = max(split_sums) if split_sums else 0.0
    rescaled_weights = {
        ir_id: exp(value - max_split_sum) for ir_id, value in log_scaled_weights.items()
    }
    split_weights = {
        gender_id: {
            rarity_id: {ir_id: rescaled_weights[ir_id] for ir_id in pool_dict}
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in merged_probs.items()
    }

    int_scale = search_scale(split_weights, rel_tol, lo=1, hi=scale_max)
    int_scale = scale_max if int_scale < 0 else min(int_scale, scale_max)

    int_weights = apply_scale(rescaled_weights, int_scale)
    return correct_zero_weights(int_weights)


def adjust_weight_settings(itemset: Data, weights: Dict[int, int]) -> None:
    existing_ids = dict.fromkeys(itemset["ItemReferenceIDs"])
    entry_list = []

    for ir_id in existing_ids:
        if ir_id in weights:
            entry_list.append((ir_id, weights[ir_id]))

    for ir_id, weight in sorted(weights.items()):
        if ir_id not in existing_ids:
            entry_list.append((ir_id, weight))

    itemset["DefaultItemWeight"] = get_value_mode(weights)
    itemset["ItemReferenceIDs"] = [ir_id for ir_id, _ in entry_list]
    itemset["AlterItemWeightMap"] = {
        str(ir_id): weight
        for ir_id, weight in entry_list
        if weight != itemset["DefaultItemWeight"]
    }


def alter_itemset_chances(itemset_node: ItemSetNode) -> None:
    altered_ir_ids = {
        ir_id
        for rarity_dicts in itemset_node.probs_to_change.values()
        for prob_dict in rarity_dicts.values()
        for ir_id in prob_dict
    }

    if not altered_ir_ids:
        return

    unchanged_probs = {
        gender_id: {
            rarity_id: itemset_node.drop_pool(
                rarity_id=rarity_id,
                gender_id=gender_id,
            )
            for rarity_id in rarity_dicts
        }
        for gender_id, rarity_dicts in itemset_node.probs_to_change.items()
    }

    logging.debug(
        "ItemSet %s current pools %s",
        itemset_node.is_id,
        json.dumps(unchanged_probs, indent=4),
    )

    merged_probs = {
        gender_id: {
            rarity_id: itemset_node.drop_pool(
                rarity_id=rarity_id,
                gender_id=gender_id,
                target_weights=pool_dict,
            )
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in itemset_node.probs_to_change.items()
    }

    logging.debug(
        "ItemSet %s probs to change %s",
        itemset_node.is_id,
        json.dumps(itemset_node.probs_to_change, indent=4),
    )

    logging.debug(
        "ItemSet %s changed pools %s",
        itemset_node.is_id,
        json.dumps(merged_probs, indent=4),
    )

    clean_merged_probs = remove_zero_prob_entries(merged_probs)

    logging.debug(
        "ItemSet %s cleaned pools %s",
        itemset_node.is_id,
        json.dumps(clean_merged_probs, indent=4),
    )

    scale_max = (1 << 31) - 1
    int_weights = get_scaled_int_weights(
        merged_probs=clean_merged_probs,
        altered_ir_ids=altered_ir_ids,
        rel_tol=itemset_node.rel_tol,
        scale_max=scale_max,
    )
    if sum(int_weights.values()) >= scale_max:
        logging.warn(
            "The item set %s weight total hit the integer limit %s. "
            "The most and least rare items may have their probabilities altered.",
            itemset_node.is_id,
            scale_max,
        )

    logging.debug(
        "ItemSet %s int weights %s",
        itemset_node.is_id,
        json.dumps(int_weights, indent=4),
    )

    int_split_probs = {
        gender_id: {
            rarity_id: normalize({ir_id: int_weights[ir_id] for ir_id in pool_dict})
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in clean_merged_probs.items()
    }
    weight_differences = {
        gender_id: {
            rarity_id: {
                ir_id: {
                    "scaled": int_split_probs[gender_id][rarity_id][ir_id],
                    "true": prob,
                }
                for ir_id, prob in pool_dict.items()
                if rel_diff(
                    inv(int_split_probs[gender_id][rarity_id][ir_id]),
                    inv(prob),
                )
                >= itemset_node.rel_tol
            }
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in clean_merged_probs.items()
    }

    logging.debug(
        "ItemSet %s weight diffs %s",
        itemset_node.is_id,
        json.dumps(weight_differences, indent=4),
    )

    logging.debug(
        "ItemSet %s final probs %s",
        itemset_node.is_id,
        json.dumps(int_split_probs, indent=4),
    )

    adjust_weight_settings(itemset_node.itemset, int_weights)


def alter_chances(
    knowledge_base: KnowledgeBase,
    config: Config,
    item_configs: List[ItemConfig],
) -> None:
    if not item_configs:
        logging.warn("No valid item configs found, dry run mode active!")

    ckb = ConfigKnowledgeBase(
        knowledge_base=knowledge_base,
        item_configs=item_configs,
        agg=(min if config.blend_using_lowest_chance else max),
        rel_tol=config.rel_tol,
    )

    inject_all_items(ckb, item_configs)

    for itemset_node in ckb.itemset_register.values():
        alter_itemset_chances(itemset_node)

    log_output_freqs(ckb, item_configs)
