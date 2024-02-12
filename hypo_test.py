import math
import random
from math import exp
from typing import Dict
from operator import itemgetter

from hypothesis import given, strategies as st


def inv(a: float, inf: float = float("inf")) -> float:
    return 1.0 / a if a > 0.0 else inf


def abs_diff(a: float, b: float, inf: float = float("inf")) -> float:
    if a == inf and b == inf:
        return 0.0
    return abs(a - b)


def rel_diff(a: float, b: float, inf: float = float("inf")) -> float:
    if a == inf and b == inf:
        return 0.0
    if b == inf:
        return 1.0
    return abs_diff(a, b) * inv(b)


def ln(a: float, ninf: float = -math.inf) -> float:
    return math.log(a) if a > 0.0 else ninf


def logsumexp(d: Dict[int, float]) -> float:
    max_val = max(d.values())
    return max_val + ln(sum([exp(val - max_val) for val in d.values()]), ninf=0.0)


def normalize(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(d.values())
    return {k: v * inv(s, inf=0.0) for k, v in d.items()}


def get_scaled_int_weights(
    merged_probs: Dict[int, Dict[int, Dict[int, float]]],
    probs_to_change: Dict[int, Dict[int, Dict[int, float]]],
    rel_tol: float,
) -> Dict[int, Dict[int, Dict[int, int]]]:
    def apply_scale(prob_dict: Dict[int, float], scale: int) -> Dict[int, int]:
        return {ir_id: int(scale * value) for ir_id, value in prob_dict.items()}

    def merge(
        acc_dict: Dict[int, float], other_dict: Dict[int, float]
    ) -> Dict[int, float]:
        cur_acc_dict = {ir_id: ln(value) for ir_id, value in acc_dict.items()}
        cur_other_dict = {ir_id: ln(value) for ir_id, value in other_dict.items()}

        shared_ir_ids = [
            ir_id
            for ir_id, acc_value in cur_acc_dict.items()
            if (
                ir_id in cur_other_dict
                and acc_value > -math.inf
                and cur_other_dict[ir_id] > -math.inf
                and acc_value < 0.0
                and cur_other_dict[ir_id] < 0.0
                and all(
                    ir_id not in pool_dict
                    for rarity_dicts in probs_to_change.values()
                    for pool_dict in rarity_dicts.values()
                )
                and rel_diff(
                    inv(exp(acc_value)),
                    inv(exp(cur_other_dict[ir_id])),
                )
                >= rel_tol
            )
        ]

        if shared_ir_ids:
            select_ir_id = shared_ir_ids[0]
            other_to_scale = cur_acc_dict[select_ir_id]
            acc_to_scale = cur_other_dict[select_ir_id]
            cur_other_dict = {
                ir_id: (
                    cur_acc_dict[ir_id]
                    if (value == -math.inf or value == 0.0) and ir_id in cur_acc_dict
                    else value + other_to_scale - acc_to_scale
                )
                for ir_id, value in cur_other_dict.items()
            }
            log_sum = logsumexp(cur_other_dict)
            cur_acc_dict = {
                ir_id: value - log_sum for ir_id, value in cur_acc_dict.items()
            }
            cur_other_dict = {
                ir_id: value - log_sum for ir_id, value in cur_other_dict.items()
            }

        exp_acc_dict = {ir_id: exp(value) for ir_id, value in cur_acc_dict.items()}
        exp_other_dict = {
            ir_id: exp(
                cur_acc_dict[ir_id]
                if (value == -math.inf or value == 0.0) and ir_id in cur_acc_dict
                else value
            )
            for ir_id, value in cur_other_dict.items()
        }

        return {**exp_acc_dict, **exp_other_dict}

    def search(
        split_probs: Dict[int, Dict[int, Dict[int, float]]], lo: int, hi: int
    ) -> int:
        if hi < lo:
            return -1

        mi = (lo + hi) // 2

        current_probs = {
            gender_id: {
                rarity_id: normalize(apply_scale(pool_dict, mi))
                for rarity_id, pool_dict in rarity_dicts.items()
            }
            for gender_id, rarity_dicts in split_probs.items()
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
            left = search(split_probs, lo, mi - 1)
            return mi if left == -1 else left

        return search(split_probs, mi + 1, hi)

    def correct_zero_weights(int_weights: Dict[int, int]) -> Dict[int, int]:
        int_scaled_weights = int_weights.copy()

        for ir_id in int_scaled_weights:
            if int_scaled_weights[ir_id] == 0:
                max_weight_ir_id, _ = max(int_scaled_weights.items(), key=itemgetter(1))
                int_scaled_weights[ir_id] = 1
                int_scaled_weights[max_weight_ir_id] -= 1

        return int_scaled_weights

    scaled_weights = {}
    for rarity_dicts in merged_probs.values():
        gender_scaled_weights = {}

        for pool_dict in rarity_dicts.values():
            gender_scaled_weights = merge(gender_scaled_weights, pool_dict)

        scaled_weights = merge(scaled_weights, gender_scaled_weights)

    split_weights = {
        gender_id: {
            rarity_id: {ir_id: scaled_weights[ir_id] for ir_id in pool_dict}
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in merged_probs.items()
    }
    split_probs = {
        gender_id: {
            rarity_id: normalize(pool_dict)
            for rarity_id, pool_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in split_weights.items()
    }

    max_split_sum = max(
        [
            sum(pool_dict.values())
            for rarity_dicts in split_weights.values()
            for pool_dict in rarity_dicts.values()
        ]
    )
    rescaled_weights = {
        ir_id: value * inv(max_split_sum, inf=0.0)
        for ir_id, value in scaled_weights.items()
    }

    scale_max = (1 << 31) - 1
    int_scale = search(split_probs, lo=1, hi=scale_max)
    if int_scale < 0:
        int_scale = scale_max

    int_weights = apply_scale(rescaled_weights, int_scale)
    return correct_zero_weights(int_weights)


def remove_zero_prob_entries(
    merged_probs: Dict[int, Dict[int, Dict[int, float]]],
) -> Dict[int, Dict[int, Dict[int, float]]]:
    zero_entries = {
        ir_id
        for rarity_dict in merged_probs.values()
        for prob_dict in rarity_dict.values()
        for ir_id in prob_dict
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


@given(
    st.dictionaries(
        keys=st.integers(min_value=1, max_value=2),
        values=st.dictionaries(
            keys=st.integers(1, 4),
            values=st.dictionaries(
                keys=st.integers(min_value=1, max_value=25),
                values=st.floats(min_value=0, max_value=1),
            ).map(
                lambda dc: {
                    k: v / sum(dc.values()) if sum(dc.values()) > 0 else 0.0
                    for k, v in dc.items()
                }
            ),
            min_size=4,
        ),
        min_size=2,
    ).map(remove_zero_prob_entries),
)
def test_get_scaled_int_weights(d: Dict[int, Dict[int, Dict[int, float]]]):
    rel_tol = 0.001
    d_filt = {
        gender_id: {
            rarity_id: {k: v for k, v in prob_dict.items() if random.random() < 0.5}
            for rarity_id, prob_dict in gender_dict.items()
        }
        for gender_id, gender_dict in d.items()
    }

    int_weights = get_scaled_int_weights(
        merged_probs=d,
        probs_to_change=d_filt,
        rel_tol=rel_tol,
    )

    split_weights = {
        gender_id: {
            rarity_id: {ir_id: int_weights[ir_id] for ir_id in rarity_dict}
            for rarity_id, rarity_dict in gender_dict.items()
        }
        for gender_id, gender_dict in d.items()
    }
    sw_sums = {
        gender_id: {
            rarity_id: sum(rarity_dict.values())
            for rarity_id, rarity_dict in gender_dict.items()
        }
        for gender_id, gender_dict in split_weights.items()
    }
    split_probs = {
        gender_id: {
            rarity_id: {
                ir_id: value * inv(sw_sums[gender_id][rarity_id], inf=0.0)
                for ir_id, value in rarity_dict.items()
            }
            for rarity_id, rarity_dict in gender_dict.items()
        }
        for gender_id, gender_dict in split_weights.items()
    }

    weight_differences = {
        gender_id: {
            rarity_id: {
                ir_id: {
                    "scaled": split_probs[gender_id][rarity_id][ir_id],
                    "true": prob,
                }
                for ir_id, prob in rarity_dict.items()
                if rel_diff(
                    inv(prob),
                    inv(split_probs[gender_id][rarity_id][ir_id]),
                )
                >= rel_tol
            }
            for rarity_id, rarity_dict in gender_dict.items()
        }
        for gender_id, gender_dict in d.items()
    }

    assert all(
        not rarity_dict
        for gender_dict in weight_differences.values()
        for rarity_dict in gender_dict.values()
    ), f"\nWD:\n    {weight_differences}\nSW:\n    {split_weights}\nDF:\n    {d}\n    {d_filt}\nIW:\n    {int_weights}"


@given(
    st.dictionaries(
        keys=st.integers(),
        values=st.floats(min_value=0, max_value=1),
    ).map(
        lambda dc: {
            k: v / sum(dc.values()) if sum(dc.values()) > 0 else 0.0
            for k, v in dc.items()
        }
    ),
    st.integers(min_value=0),
)
def test_apply_scale(prob_dict: Dict[int, float], scale: int):
    scaled_dict = apply_scale(prob_dict, scale)
    assert sum(scaled_dict.values()) <= scale


def apply_scale(prob_dict: Dict[int, float], scale: int) -> Dict[int, int]:
    return {ir_id: int(scale * value) for ir_id, value in prob_dict.items()}


if __name__ == "__main__":
    test_get_scaled_int_weights()
