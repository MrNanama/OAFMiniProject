"""Active / retired strategies. Each module oafbt/strategies/<name>.py defines STRATEGY = <class>."""
import importlib

ACTIVE = [
    "s01_gp_trend",
    "s02_factor_momentum",
    "s03_pca_statarb",
    "s04_volume_attention",
    "s05_principal_portfolios",
    "s06_curve_pca_rv",
    "s07_industry_seasonality",
    "s08_ml_stock_ranker",
    "s09_vrp_timing",
    "s10_turn_of_month",
]
RETIRED = []   # (name, round retired, reason)


def load(name: str, **params):
    mod = importlib.import_module(f"oafbt.strategies.{name}")
    importlib.reload(mod)
    return mod.STRATEGY(**params)
