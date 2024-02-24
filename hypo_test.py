import random
from typing import Dict

from hypothesis import given, strategies as st

from rarify.math import inv, rel_diff, normalize
from rarify.solver import remove_zero_prob_entries, get_scaled_int_weights, apply_scale


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
    scale_max = (1 << 31) - 1
    d_filt = {
        k
        for rarity_dicts in d.values()
        for prob_dict in rarity_dicts.values()
        for k in prob_dict
        if random.random() < 0.5
    }

    int_weights = get_scaled_int_weights(
        merged_probs=d,
        altered_ir_ids=d_filt,
        rel_tol=rel_tol,
        scale_max=scale_max,
    )

    split_weights = {
        gender_id: {
            rarity_id: {ir_id: int_weights[ir_id] for ir_id in rarity_dict}
            for rarity_id, rarity_dict in gender_dict.items()
        }
        for gender_id, gender_dict in d.items()
    }
    split_probs = {
        gender_id: {
            rarity_id: normalize(prob_dict)
            for rarity_id, prob_dict in rarity_dicts.items()
        }
        for gender_id, rarity_dicts in split_weights.items()
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
        not diff_dict
        for rarity_dicts in weight_differences.values()
        for diff_dict in rarity_dicts.values()
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


if __name__ == "__main__":
    test_get_scaled_int_weights()
    # test_apply_scale()
