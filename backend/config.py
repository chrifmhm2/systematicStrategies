from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    debug: bool = False
    default_data_source: str = "yahoo"
    max_backtest_years: int = 10


settings = Settings()
