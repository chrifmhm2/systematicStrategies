from dataclasses import dataclass
from typing import Literal


@dataclass
class VanillaOption:
    """A plain European call or put on a single underlying."""

    underlying: str
    strike: float
    maturity: float               # time to expiry in years
    option_type: Literal["call", "put"] = "call"


@dataclass
class BasketOption:
    """
    A European call on a weighted basket of underlyings.

    Payoff = max(Σ ω_i * S_i^T − K, 0)

    `underlyings` and `weights` must have the same length,
    and weights should sum to 1.
    """

    underlyings: list[str]
    weights: list[float]          # ω_i, should sum to 1
    strike: float                 # K
    maturity: float               # T in years
    option_type: Literal["call", "put"] = "call"
