"""Active / retired strategies. Each module oafbt/strategies/<name>.py defines STRATEGY = <class>."""
import importlib

ACTIVE = [
    "s01_gp_trend",
    "s02_factor_momentum",
    "s05_principal_portfolios",
    "s07_industry_seasonality",
    "s08_ml_stock_ranker",
    "s09_vrp_timing",
    "s10_turn_of_month",
    "s11_cross_asset_carry",
    "s12_crypto_trend",
    "s14_bab",
]
RETIRED = [  # (name, round retired, reason)
    ("s06_curve_pca_rv", 1, "2%-vol ETF butterfly cannot pay ETF costs + short borrow at a 3x leverage cap (SR -0.49)"),
    ("s04_volume_attention", 1, "no gross premium in S&P 500 large caps: high-volume premium (v1 -0.26) and expected-"
                                "attention / earnings-cycle volume seasonality (v2 -0.08)"),
    ("s03_pca_statarb", 1, "gross edge ~0.7 SR but per-round-trip edge <= costs after 2008 (v1 -0.02, v2 +0.02)"),
    ("s13_residual_momentum", 1, "candidate replacement rejected: SR -0.14 (2004-20)"),
]


def load(name: str, **params):
    mod = importlib.import_module(f"oafbt.strategies.{name}")
    importlib.reload(mod)
    return mod.STRATEGY(**params)
