import math
from typing import Any, Dict
from collections import Counter


INF = math.inf
NINF = -math.inf


def inv(a: float, inf: float = INF) -> float:
    return 1.0 / a if a > 0.0 else inf


def abs_diff(a: float, b: float, inf: float = INF) -> float:
    if a == inf and b == inf:
        return 0.0
    return abs(a - b)


def rel_diff(a: float, b: float, inf: float = INF) -> float:
    if a == inf and b == inf:
        return 0.0
    if b == inf:
        return 1.0
    return abs_diff(a, b) * inv(b)


def normalize(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(d.values())
    return {k: v * inv(s, inf=0.0) for k, v in d.items()}


def get_value_mode(dct: Dict[Any, float]) -> float:
    return Counter(dct.values()).most_common(1)[0][0]


def exp(a: float) -> float:
    return math.exp(a)


def ln(a: float, ninf: float = NINF) -> float:
    return math.log(a) if a > 0.0 else ninf


def logsumexp(d: Dict[int, float]) -> float:
    max_val = max(d.values())
    return max_val + ln(sum([exp(val - max_val) for val in d.values()]), ninf=0.0)
