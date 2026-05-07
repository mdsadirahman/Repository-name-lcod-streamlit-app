# ============================================================
# INTERACTIVE LCOD STREAMLIT APP
# One-mode MDPI-style version:
#   1) No separate MDPI/Interactive modes in Streamlit
#   2) Always runs all applications in fixed original order
#   3) CNG is included for Refuse, Transit Bus, Drayage, and Long Haul
#   4) Drayage/Long Haul CNG are trial values copied from diesel inputs
#   5) Sidebar inputs are grouped by selected application and vehicle type
#   6) Breakeven uses exact diesel LCOD from same mother LCOD run
# ============================================================

import copy
import random
from typing import Dict, Tuple, Any, Literal
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Interactive LCOD Model", layout="wide")

st.title("Interactive LCOD Model")
st.markdown("One-mode MDPI-style LCOD and breakeven analysis for Refuse, Transit Bus, Drayage, and Long Haul. CNG is included for all applications.")

# ============================================================
# PLOT STYLE
# ============================================================
FONT_XTICK = 14
FONT_YTICK = 14
FONT_AXES_LABEL = 14
FONT_AXES_TITLE = 15
FONT_LEGEND = 12
FONT_FIGURE = 18

plt.rcParams.update({
    "xtick.labelsize": FONT_XTICK,
    "ytick.labelsize": FONT_YTICK,
    "axes.labelsize": FONT_AXES_LABEL,
    "axes.titlesize": FONT_AXES_TITLE,
    "legend.fontsize": FONT_LEGEND,
    "figure.titlesize": FONT_FIGURE,
})

# ============================================================
# TYPES AND CONSTANTS
# ============================================================
Scenario = Literal["current", "2030"]
Range = Tuple[float, float]

DEFAULT_N_SAMPLES = 20_000
DEFAULT_RANDOM_SEED = 7
PCTILES = (5, 50, 95)

APP_ORDER = ["refuse", "bus", "drayage", "longhaul"]
VEHICLE_ORDER = ["diesel", "fcev", "bev", "cng"]

VEHICLE_COLORS = {
    "diesel": "#4D4D4D",
    "fcev": "#1F77B4",
    "bev": "#2CA02C",
    "cng": "#FF7F0E",
}

REQUIRED_RES_KEYS = [
    "initial_cost_current",
    "initial_cost_2030",
    "residual_factor_current",
    "residual_factor_2030",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def _is_range(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)

def _sample_uniform(rng: random.Random, r: Range) -> float:
    lo, hi = r
    if lo > hi:
        raise ValueError(f"Bad range {r}: lo > hi")
    return lo if lo == hi else lo + (hi - lo) * rng.random()

def _summarize_percentiles(arr: np.ndarray, pctiles=(5, 50, 95)) -> Dict[str, float]:
    return {f"p{p}": float(np.percentile(arr, p)) for p in pctiles}

def _as_range(x: Range | float | int) -> Range:
    if isinstance(x, (int, float)):
        return (float(x), float(x))
    return (float(x[0]), float(x[1]))

def _cost_range(power_kw: Range | float, price_per_kw: Range | float, max_mult: float = 1.0) -> Range:
    p_lo, p_hi = _as_range(power_kw)
    c_lo, c_hi = _as_range(price_per_kw)
    return (float(p_lo * c_lo), float(p_hi * c_hi * float(max_mult)))

def _h2_storage_cost_range(h2_storage_required: Range | float, max_mult: float = 1.0) -> Range:
    s_lo, s_hi = _as_range(h2_storage_required)
    k = 12.7 * 33.1
    return (float(k * s_lo), float(k * s_hi * float(max_mult)))

def _vehicle_pretty(vt: str) -> str:
    return vt.upper()

def bev_battery_mass_kg(battery_energy_kwh: float, energy_density_kwh_per_kg: float) -> float:
    if energy_density_kwh_per_kg <= 0:
        raise ValueError("BEV energy density must be > 0.")
    return battery_energy_kwh / energy_density_kwh_per_kg

def bev_revenue_weight_ton(rev_weight_ton_diesel_fcev: float, battery_mass_kg: float) -> float:
    return rev_weight_ton_diesel_fcev - battery_mass_kg / 1000.0

def residual_cost_usd_point(residual_tbl: Dict[str, Any], vehicle_type: str, scenario: Scenario) -> float:
    comps = residual_tbl[vehicle_type]
    tot = 0.0
    for comp in comps.values():
        if scenario == "current":
            tot += comp["initial_cost_current"] * comp["residual_factor_current"]
        else:
            tot += comp["initial_cost_2030"] * comp["residual_factor_2030"]
    return float(tot)

def lcod_usd_per_ton_mile_point(lcod_mile: float, revenue_weight_ton: float) -> float:
    if revenue_weight_ton <= 0:
        raise ValueError("Revenue weight must be > 0.")
    return float(lcod_mile / revenue_weight_ton)

def range_input(label: str, default_range: Range, key: str) -> Range:
    col1, col2 = st.columns(2)
    with col1:
        lo = st.number_input(f"{label} min", value=float(default_range[0]), key=f"{key}_lo")
    with col2:
        hi = st.number_input(f"{label} max", value=float(default_range[1]), key=f"{key}_hi")
    return (lo, hi)

# ============================================================
# BASE APPLICATION DATA
# ============================================================
def build_base_applications() -> Dict[str, Any]:

    applications = {
        # --------------------------------------------------------
        # REFUSE
        # --------------------------------------------------------
        "refuse": {
            "label": "Refuse",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (300.0, 310.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (319000.0, 355000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.8),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (445000.0, 540000.0),
                    "fuel_economy_mi_per_unit": (5.0, 6.0),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (438575.0, 671000.0),
                    "fuel_economy_mi_per_unit": (0.30, 0.32),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "cng": {
                    "purchase_cost_usd": (406000.0, 485000.0),
                    "fuel_economy_mi_per_unit": (1.90, 2.40),
                    "fuel_price_usd_per_unit": (2.90, 3.20),
                    "planned_miles_per_year": (25000.0, 26000.0),
                    "maintenance_usd_per_mile": (0.72, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # TRANSIT BUS
        # --------------------------------------------------------
        "bus": {
            "label": "Transit Bus",
            "price_mode": "base_plus_premium",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (300.0, 310.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "base_price_usd": (440000.0, 450000.0),
                    "premium_price_usd": (0.0, 0.0),
                    "fuel_economy_mi_per_unit": (3.0, 4.0),
                    "fuel_price_usd_per_unit": (3.5, 4.0),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "base_price_usd": (385455.0, 411438.0),
                    "premium_price_usd": (350000.0, 738562.0),
                    "fuel_economy_mi_per_unit": (6.5, 7.5),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "base_price_usd": (385000.0, 514585.0),
                    "premium_price_usd": (350000.0, 585415.0),
                    "fuel_economy_mi_per_unit": (0.34, 0.40),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "cng": {
                    "base_price_usd": (600000.0, 650000.0),
                    "premium_price_usd": (0.0, 0.0),
                    "fuel_economy_mi_per_unit": (2.80, 3.00),
                    "fuel_price_usd_per_unit": (2.90, 3.20),
                    "planned_miles_per_year": (40000.0, 43000.0),
                    "maintenance_usd_per_mile": (0.72, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # DRAYAGE
        # --------------------------------------------------------
        "drayage": {
            "label": "Drayage",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (180.0, 378.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (151000.0, 178000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.2),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (348000.0, 385000.0),
                    "fuel_economy_mi_per_unit": (5.1, 6.9),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (233575.0, 388000.0),
                    "fuel_economy_mi_per_unit": (0.27, 0.30),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                # Trial CNG values copied from diesel for first-pass comparison.
                # Replace these with literature CNG parameters when finalized.
                "cng": {
                    "purchase_cost_usd": (151000.0, 178000.0),
                    "fuel_economy_mi_per_unit": (2.0, 2.2),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (14000.0, 15000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
            },
        },

        # --------------------------------------------------------
        # LONG HAUL
        # --------------------------------------------------------
        "longhaul": {
            "label": "Long Haul",
            "price_mode": "single_purchase",
            "GLOBAL_R": {
                "lifetime_years": (10.0, 12.0),
                "revenue_weight_ton_diesel_fcev": (22.79, 22.79),
                "bev_battery_energy_kwh": (650.0, 850.0),
                "bev_energy_density_kwh_per_kg": (0.175, 0.175),
            },
            "VEH_R": {
                "diesel": {
                    "purchase_cost_usd": (171000.0, 210000.0),
                    "fuel_economy_mi_per_unit": (6.0, 7.0),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
                "fcev": {
                    "purchase_cost_usd": (430000.0, 489000.0),
                    "fuel_economy_mi_per_unit": (7.0, 8.1),
                    "fuel_price_usd_per_unit": (6.50, 7.00),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                "bev": {
                    "purchase_cost_usd": (410575.0, 458000.0),
                    "fuel_economy_mi_per_unit": (0.43, 0.59),
                    "fuel_price_usd_per_unit": (0.40, 0.60),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.55, 0.708),
                },
                # Trial CNG values copied from diesel for first-pass comparison.
                # Replace these with literature CNG parameters when finalized.
                "cng": {
                    "purchase_cost_usd": (171000.0, 210000.0),
                    "fuel_economy_mi_per_unit": (6.0, 7.0),
                    "fuel_price_usd_per_unit": (3.0, 4.0),
                    "planned_miles_per_year": (65000.0, 65000.0),
                    "maintenance_usd_per_mile": (0.45, 0.943),
                },
            },
        },
    }

    # Drayage and Long Haul CNG are already included above as diesel-like trial values.

    return applications

# ============================================================
# ADD RESIDUAL TABLES
# ============================================================
def add_residual_tables(applications: Dict[str, Any]) -> Dict[str, Any]:

    applications["refuse"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(237000.0, 237000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((200.0, 250.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((80.0, 100.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((130.0, 130.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((25.0, 25.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["bus"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(293000.0, 300000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((384.0, 400.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((80.0, 80.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((85.0, 100.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((60.0, 70.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(400000.0, 433000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["drayage"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(119000.0, 119000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((180.0, 378.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((70.0, 160.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((168.0, 210.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((26.0, 40.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    applications["longhaul"]["RESIDUAL_R"] = {
        "diesel": {
            "overall": dict(
                initial_cost_current=(134000.0, 134000.0),
                initial_cost_2030=(237000.0, 237000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "bev": {
            "battery": dict(
                initial_cost_current=_cost_range((650.0, 850.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((100.0, 100.0), (90.0, 90.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "motor": dict(
                initial_cost_current=(9969.0, 9969.0),
                initial_cost_2030=(6935.0, 6935.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
        "fcev": {
            "battery": dict(
                initial_cost_current=_cost_range((200.0, 300.0), (108.0, 175.0), 1.00),
                initial_cost_2030=_cost_range((80.0, 80.0), (95.0, 95.0), 1.00),
                residual_factor_current=(0.43, 0.43),
                residual_factor_2030=(0.49, 0.49),
            ),
            "fuel_cell": dict(
                initial_cost_current=_cost_range((318.0, 350.0), (300.0, 300.0), 1.00),
                initial_cost_2030=_cost_range((180.0, 180.0), (650.0, 650.0), 1.00),
                residual_factor_current=(0.25, 0.25),
                residual_factor_2030=(0.25, 0.25),
            ),
            "motor": dict(
                initial_cost_current=(11136.0, 11136.0),
                initial_cost_2030=(7416.0, 7416.0),
                residual_factor_current=(0.35, 0.35),
                residual_factor_2030=(0.35, 0.35),
            ),
            "glider": dict(
                initial_cost_current=(75000.0, 75000.0),
                initial_cost_2030=(82000.0, 82000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
            "hydrogen_tank": dict(
                initial_cost_current=_h2_storage_cost_range((60.0, 75.0), 1.00),
                initial_cost_2030=_h2_storage_cost_range((40.0, 40.0), 1.00),
                residual_factor_current=(0.70, 0.70),
                residual_factor_2030=(0.70, 0.70),
            ),
        },
        "cng": {
            "overall": dict(
                initial_cost_current=(274000.0, 274000.0),
                initial_cost_2030=(274000.0, 274000.0),
                residual_factor_current=(0.15, 0.15),
                residual_factor_2030=(0.15, 0.15),
            ),
        },
    }

    return applications

# ============================================================
# VEHICLE COMPLETENESS CHECK
# ============================================================
def vehicle_is_complete(app_cfg: Dict[str, Any], vt: str) -> bool:
    VEH_R = app_cfg["VEH_R"]
    RESIDUAL_R = app_cfg["RESIDUAL_R"]
    price_mode = app_cfg["price_mode"]

    if vt not in VEH_R or vt not in RESIDUAL_R:
        return False

    v = VEH_R[vt]
    if not isinstance(v, dict):
        return False

    if price_mode == "single_purchase":
        req_keys = [
            "purchase_cost_usd",
            "fuel_economy_mi_per_unit",
            "fuel_price_usd_per_unit",
            "planned_miles_per_year",
            "maintenance_usd_per_mile",
        ]
    else:
        req_keys = [
            "base_price_usd",
            "premium_price_usd",
            "fuel_economy_mi_per_unit",
            "fuel_price_usd_per_unit",
            "planned_miles_per_year",
            "maintenance_usd_per_mile",
        ]

    for k in req_keys:
        if k not in v or v[k] is None or not _is_range(v[k]):
            return False

    for _, fields in RESIDUAL_R[vt].items():
        for rk in REQUIRED_RES_KEYS:
            if rk not in fields or fields[rk] is None or not _is_range(fields[rk]):
                return False

    return True

# ============================================================
# LCOD COMPONENT FUNCTIONS
# ============================================================
def lcod_components_per_mile_single_purchase(v, g, residual_cost):
    fe = v["fuel_economy_mi_per_unit"]
    if fe <= 0:
        raise ValueError("Fuel economy must be > 0.")

    lifetime_miles = g["lifetime_years"] * v["planned_miles_per_year"]
    if lifetime_miles <= 0:
        raise ValueError("Lifetime miles must be > 0.")

    fuel_per_mile = (1.0 / fe) * v["fuel_price_usd_per_unit"]
    purchase_per_mile = (v["purchase_cost_usd"] - residual_cost) / lifetime_miles
    maintenance_per_mile = v["maintenance_usd_per_mile"]
    total = fuel_per_mile + purchase_per_mile + maintenance_per_mile

    return {
        "fuel": float(fuel_per_mile),
        "purchase": float(purchase_per_mile),
        "maintenance": float(maintenance_per_mile),
        "total": float(total),
        "lifetime_miles": float(lifetime_miles),
        "purchase_total": float(v["purchase_cost_usd"]),
    }

def lcod_components_per_mile_base_premium(v, g, residual_cost):
    fe = v["fuel_economy_mi_per_unit"]
    if fe <= 0:
        raise ValueError("Fuel economy must be > 0.")

    lifetime_miles = g["lifetime_years"] * v["planned_miles_per_year"]
    if lifetime_miles <= 0:
        raise ValueError("Lifetime miles must be > 0.")

    fuel_per_mile = (1.0 / fe) * v["fuel_price_usd_per_unit"]
    purchase_total = v["base_price_usd"] + v["premium_price_usd"]
    purchase_per_mile = (purchase_total - residual_cost) / lifetime_miles
    maintenance_per_mile = v["maintenance_usd_per_mile"]
    total = fuel_per_mile + purchase_per_mile + maintenance_per_mile

    return {
        "fuel": float(fuel_per_mile),
        "purchase": float(purchase_per_mile),
        "maintenance": float(maintenance_per_mile),
        "total": float(total),
        "lifetime_miles": float(lifetime_miles),
        "purchase_total": float(purchase_total),
    }

# ============================================================
# RUN APPLICATION MODEL
# IMPORTANT:
#   rng is created inside each application, same as original mother code.
#   veh_list is fixed by original vehicle order and completeness.
# ============================================================
def run_application_model(app_name: str, app_cfg: Dict[str, Any], n_samples: int, random_seed: int) -> Dict[str, Any]:

    GLOBAL_R = app_cfg["GLOBAL_R"]
    VEH_R = app_cfg["VEH_R"]
    RESIDUAL_R = app_cfg["RESIDUAL_R"]
    price_mode = app_cfg["price_mode"]

    veh_list = [vt for vt in VEHICLE_ORDER if vehicle_is_complete(app_cfg, vt)]

    rng = random.Random(int(random_seed))
    N = int(n_samples)

    lcod_mile = {vt: [] for vt in veh_list}
    lcod_tm = {vt: [] for vt in veh_list}

    breakeven_inputs = {
        vt: {
            "fuel_per_mile": [],
            "maintenance_per_mile": [],
            "lifetime_miles": [],
            "residual_cost": [],
            "purchase_total": [],
        }
        for vt in veh_list
    }

    for _ in range(N):
        g = {k: _sample_uniform(rng, r) for k, r in GLOBAL_R.items()}

        batt_mass = bev_battery_mass_kg(
            g["bev_battery_energy_kwh"],
            g["bev_energy_density_kwh_per_kg"]
        )

        rev_bev = bev_revenue_weight_ton(
            g["revenue_weight_ton_diesel_fcev"],
            batt_mass
        )

        for vt in veh_list:
            v = {
                k: _sample_uniform(rng, rv)
                for k, rv in VEH_R[vt].items()
                if rv is not None and _is_range(rv)
            }

            residual_tbl = {vt: {}}

            for cname, fields in RESIDUAL_R[vt].items():
                residual_tbl[vt][cname] = {
                    rk: _sample_uniform(rng, fields[rk])
                    for rk in REQUIRED_RES_KEYS
                }

            residual_cost = residual_cost_usd_point(residual_tbl, vt, "current")

            if price_mode == "single_purchase":
                comp = lcod_components_per_mile_single_purchase(v, g, residual_cost)
            else:
                comp = lcod_components_per_mile_base_premium(v, g, residual_cost)

            rev_wt = rev_bev if vt == "bev" else g["revenue_weight_ton_diesel_fcev"]

            lmi = comp["total"]
            ltm = lcod_usd_per_ton_mile_point(lmi, rev_wt)

            lcod_mile[vt].append(lmi)
            lcod_tm[vt].append(ltm)

            breakeven_inputs[vt]["fuel_per_mile"].append(comp["fuel"])
            breakeven_inputs[vt]["maintenance_per_mile"].append(comp["maintenance"])
            breakeven_inputs[vt]["lifetime_miles"].append(comp["lifetime_miles"])
            breakeven_inputs[vt]["residual_cost"].append(residual_cost)
            breakeven_inputs[vt]["purchase_total"].append(comp["purchase_total"])

    return {
        "application": app_name,
        "label": app_cfg["label"],
        "VEH_LIST": veh_list,
        "lcod_mile": lcod_mile,
        "lcod_tm": lcod_tm,
        "breakeven_inputs": breakeven_inputs,
    }

# ============================================================
# STREAMLIT CONTROLS
# ============================================================
st.sidebar.header("Run Settings")

# ============================================================
# STREAMLIT CONTROLS + FORM
# ============================================================

# One MDPI-style workflow only: CNG is included for all applications.
APPLICATIONS = build_base_applications()
APPLICATIONS = add_residual_tables(APPLICATIONS)

with st.sidebar.form("input_form"):

    st.header("Run Settings")

    N_SAMPLES = st.slider(
        "Monte Carlo samples",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_N_SAMPLES,
        step=1000,
    )

    RANDOM_SEED = st.number_input(
        "Random seed",
        value=DEFAULT_RANDOM_SEED,
        step=1,
    )

    st.markdown("---")
    st.header("Display Options")

    selected_apps = st.multiselect(
        "Applications to display",
        options=APP_ORDER,
        default=APP_ORDER,
        format_func=lambda x: APPLICATIONS[x]["label"],
    )

    selected_vehicles_display = st.multiselect(
        "Vehicles to display",
        options=VEHICLE_ORDER,
        default=VEHICLE_ORDER,
        format_func=lambda x: x.upper(),
    )

    # ========================================================
    # INPUT EDITOR
    # ========================================================

    st.markdown("---")
    st.header("Edit Input Ranges")

    edit_app_key = st.selectbox(
        "Application to edit",
        options=APP_ORDER,
        format_func=lambda x: APPLICATIONS[x]["label"],
    )

    edit_app = APPLICATIONS[edit_app_key]

    with st.expander(f"{edit_app['label']} — Global Inputs", expanded=False):

        for k, r in edit_app["GLOBAL_R"].items():

            APPLICATIONS[edit_app_key]["GLOBAL_R"][k] = range_input(
                k,
                r,
                f"{edit_app_key}_global_{k}"
            )

    for vt in VEHICLE_ORDER:

        if not vehicle_is_complete(APPLICATIONS[edit_app_key], vt):
            continue

        with st.expander(f"{vt.upper()} Inputs", expanded=False):

            st.markdown("### Vehicle Inputs")

            for k, r in edit_app["VEH_R"][vt].items():

                APPLICATIONS[edit_app_key]["VEH_R"][vt][k] = range_input(
                    k,
                    r,
                    f"{edit_app_key}_{vt}_{k}"
                )

            st.markdown("### Residual Inputs")

            for cname, fields in edit_app["RESIDUAL_R"][vt].items():

                st.markdown(f"**{cname}**")

                for k, r in fields.items():

                    APPLICATIONS[edit_app_key]["RESIDUAL_R"][vt][cname][k] = range_input(
                        k,
                        r,
                        f"{edit_app_key}_{vt}_{cname}_{k}"
                    )

    st.markdown("---")

    run_button = st.form_submit_button(
        "Run Model",
        use_container_width=True
    )

# ============================================================
# WAIT UNTIL BUTTON IS CLICKED
# ============================================================

if "all_results" not in st.session_state:
    st.info("Adjust sidebar inputs and click 'Run Model' to calculate results.")
    st.stop()

# ============================================================
# BASIC CHECKS
# ============================================================

if len(selected_apps) == 0 or len(selected_vehicles_display) == 0:

    st.warning(
        "Select at least one application and one vehicle."
    )

    st.stop()
if run_button:
    all_results = {}

    try:
        for app_key in APP_ORDER:
            all_results[app_key] = run_application_model(
                app_key,
                APPLICATIONS[app_key],
                n_samples=N_SAMPLES,
                random_seed=RANDOM_SEED
            )

        st.session_state["all_results"] = all_results
        st.session_state["APPLICATIONS"] = APPLICATIONS
        st.session_state["selected_apps"] = selected_apps
        st.session_state["selected_vehicles_display"] = selected_vehicles_display

    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

else:
    all_results = st.session_state["all_results"]
    APPLICATIONS = st.session_state["APPLICATIONS"]
    selected_apps = st.session_state["selected_apps"]
    selected_vehicles_display = st.session_state["selected_vehicles_display"]

st.success(
    "One-mode MDPI-style run is active. The model runs all applications in fixed original order, "
    "CNG is included for all applications, and breakeven uses diesel LCOD from the same run."
)

# ============================================================
# GROUPED SUMMARY TABLES
# ============================================================
st.subheader("Grouped LCOD Summary Tables")

all_rows = []

for app_key in selected_apps:
    res = all_results[app_key]
    rows = []

    for vt in res["VEH_LIST"]:
        if vt not in selected_vehicles_display:
            continue

        mile_s = _summarize_percentiles(np.array(res["lcod_mile"][vt], dtype=float), PCTILES)
        tm_s = _summarize_percentiles(np.array(res["lcod_tm"][vt], dtype=float), PCTILES)

        row = {
            "Vehicle": vt.upper(),
            "P5 ($/mile)": mile_s["p5"],
            "P50 ($/mile)": mile_s["p50"],
            "P95 ($/mile)": mile_s["p95"],
            "P5 ($/ton-mile)": tm_s["p5"],
            "P50 ($/ton-mile)": tm_s["p50"],
            "P95 ($/ton-mile)": tm_s["p95"],
        }

        rows.append(row)
        all_rows.append({"Application": res["label"], **row})

    st.markdown(f"### {res['label']}")

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No selected vehicles available for this application.")

summary_df = pd.DataFrame(all_rows)

if not summary_df.empty:
    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download grouped LCOD tables as CSV",
        data=csv,
        file_name="grouped_lcod_summary.csv",
        mime="text/csv"
    )

# ============================================================
# PLOT FUNCTION
# ============================================================
def make_lcod_plot(metric_key: str, ylabel: str, title_prefix: str):
    apps_to_plot = selected_apps

    fig, axes = plt.subplots(
        1,
        len(apps_to_plot),
        figsize=(6 * len(apps_to_plot), 5.5)
    )

    if len(apps_to_plot) == 1:
        axes = [axes]

    for ax, app_key in zip(axes, apps_to_plot):
        res = all_results[app_key]

        vehs = [
            vt for vt in res["VEH_LIST"]
            if vt in selected_vehicles_display
        ]

        if len(vehs) == 0:
            ax.axis("off")
            ax.set_title(res["label"])
            continue

        x = np.arange(len(vehs))

        p5 = []
        p50 = []
        p95 = []

        for vt in vehs:
            s = _summarize_percentiles(np.array(res[metric_key][vt], dtype=float), PCTILES)
            p5.append(s["p5"])
            p50.append(s["p50"])
            p95.append(s["p95"])

        p5 = np.array(p5)
        p50 = np.array(p50)
        p95 = np.array(p95)

        ax.bar(x, p50, width=0.55, color=[VEHICLE_COLORS[v] for v in vehs], edgecolor="black", alpha=0.85)
        ax.errorbar(
            x,
            p50,
            yerr=[p50 - p5, p95 - p50],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1.2
        )
        ax.scatter(x, p50, color="black", s=35, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([_vehicle_pretty(v) for v in vehs])
        ax.set_ylabel(ylabel)
        ax.set_title(res["label"])
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig

# ============================================================
# LCOD PLOTS
# ============================================================
st.subheader("LCOD Plot: $/ton-mile")

fig_tm = make_lcod_plot(
    metric_key="lcod_tm",
    ylabel="LCOD ($/ton-mile)",
    title_prefix="LCOD $/ton-mile"
)

st.pyplot(fig_tm)

buf_tm = BytesIO()
fig_tm.savefig(buf_tm, format="png", dpi=600, bbox_inches="tight")
buf_tm.seek(0)

st.download_button(
    label="Download $/ton-mile plot PNG, 600 dpi",
    data=buf_tm,
    file_name="lcod_ton_mile.png",
    mime="image/png"
)

st.subheader("LCOD Plot: $/mile")

fig_mile = make_lcod_plot(
    metric_key="lcod_mile",
    ylabel="LCOD ($/mile)",
    title_prefix="LCOD $/mile"
)

st.pyplot(fig_mile)

buf_mile = BytesIO()
fig_mile.savefig(buf_mile, format="png", dpi=600, bbox_inches="tight")
buf_mile.seek(0)

st.download_button(
    label="Download $/mile plot PNG, 600 dpi",
    data=buf_mile,
    file_name="lcod_mile.png",
    mime="image/png"
)

# ============================================================
# BREAKEVEN ANALYSIS
# ============================================================
st.subheader("Breakeven Analysis vs Diesel")

st.markdown(
    "Breakeven purchase price is calculated using the same diesel LCOD distribution "
    "from the mother LCOD run. Diesel is not resampled."
)

available_be_apps = [
    app_key for app_key in selected_apps
    if "diesel" in all_results[app_key]["VEH_LIST"]
]

if len(available_be_apps) == 0:
    st.warning("Diesel must be available to calculate breakeven.")
else:
    be_col1, be_col2 = st.columns(2)

    with be_col1:
        be_app = st.selectbox(
            "Select application for breakeven",
            options=available_be_apps,
            format_func=lambda x: APPLICATIONS[x]["label"]
        )

    available_alt = [
        vt for vt in all_results[be_app]["VEH_LIST"]
        if vt != "diesel"
    ]

    with be_col2:
        be_vehicle = st.selectbox(
            "Select alternative vehicle",
            options=available_alt,
            format_func=lambda x: x.upper()
        )

    res = all_results[be_app]

    diesel_lcod = np.array(res["lcod_mile"]["diesel"], dtype=float)

    alt_fuel = np.array(res["breakeven_inputs"][be_vehicle]["fuel_per_mile"], dtype=float)
    alt_maint = np.array(res["breakeven_inputs"][be_vehicle]["maintenance_per_mile"], dtype=float)
    alt_lifetime_miles = np.array(res["breakeven_inputs"][be_vehicle]["lifetime_miles"], dtype=float)
    alt_residual = np.array(res["breakeven_inputs"][be_vehicle]["residual_cost"], dtype=float)
    alt_current_purchase = np.array(res["breakeven_inputs"][be_vehicle]["purchase_total"], dtype=float)

    purchase_breakeven = (
        (diesel_lcod - alt_fuel - alt_maint) * alt_lifetime_miles
        + alt_residual
    )

    valid_mask = np.isfinite(purchase_breakeven)
    purchase_breakeven = purchase_breakeven[valid_mask]
    alt_current_purchase = alt_current_purchase[valid_mask]

    be_p5 = np.percentile(purchase_breakeven, 5)
    be_p50 = np.percentile(purchase_breakeven, 50)
    be_p95 = np.percentile(purchase_breakeven, 95)

    current_p5 = np.percentile(alt_current_purchase, 5)
    current_p50 = np.percentile(alt_current_purchase, 50)
    current_p95 = np.percentile(alt_current_purchase, 95)

    gap_p50 = current_p50 - be_p50

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Breakeven P50 purchase price", f"${be_p50:,.0f}")
    c2.metric("Current P50 purchase price", f"${current_p50:,.0f}")
    c3.metric("P50 gap: Current - Breakeven", f"${gap_p50:,.0f}")
    c4.metric("Breakeven P5–P95", f"${be_p5:,.0f}–${be_p95:,.0f}")

    be_table = pd.DataFrame([
        {
            "Application": res["label"],
            "Vehicle": be_vehicle.upper(),
            "Breakeven P5 ($)": be_p5,
            "Breakeven P50 ($)": be_p50,
            "Breakeven P95 ($)": be_p95,
            "Current Purchase P5 ($)": current_p5,
            "Current Purchase P50 ($)": current_p50,
            "Current Purchase P95 ($)": current_p95,
            "P50 Gap: Current - Breakeven ($)": gap_p50,
        }
    ])

    st.markdown("### Breakeven Summary Table")
    st.dataframe(be_table, use_container_width=True)

    # ========================================================
    # NORMAL DISTRIBUTION PLOT
    # ========================================================
    st.markdown("### Breakeven Purchase Price Distribution")

    mu = np.mean(purchase_breakeven)
    sigma = np.std(purchase_breakeven, ddof=1)

    fig_be, ax = plt.subplots(figsize=(8, 5))

    alt_color = VEHICLE_COLORS.get(be_vehicle, "#808080")

    ax.hist(
        purchase_breakeven,
        bins=40,
        density=True,
        color=alt_color,
        edgecolor="black",
        alpha=0.45,
        label="Breakeven distribution"
    )

    if sigma > 0:
        x_grid = np.linspace(
            np.min(purchase_breakeven),
            np.max(purchase_breakeven),
            500
        )

        normal_pdf = (
            1.0 / (sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
        )

        ax.plot(
            x_grid,
            normal_pdf,
            color=alt_color,
            linewidth=2.5,
            label="Normal fit"
        )

    ax.axvline(be_p50, color="black", linestyle="--", linewidth=2.0, label="Breakeven P50")
    ax.axvline(current_p50, color=alt_color, linestyle="-.", linewidth=2.0, label="Current purchase P50")
    ax.axvspan(be_p5, be_p95, color=alt_color, alpha=0.12, label="Breakeven P5–P95")

    ax.set_xlabel("Required breakeven purchase price ($)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{res['label']} — {be_vehicle.upper()} vs DIESEL")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    st.pyplot(fig_be)

    buf_be = BytesIO()
    fig_be.savefig(buf_be, format="png", dpi=600, bbox_inches="tight")
    buf_be.seek(0)

    st.download_button(
        label="Download breakeven distribution plot PNG, 600 dpi",
        data=buf_be,
        file_name=f"breakeven_{be_app}_{be_vehicle}.png",
        mime="image/png"
    )

# ============================================================
# FINAL NOTE
# ============================================================
st.info(
    "Changing the Monte Carlo sample number, random seed, or sidebar input ranges automatically reruns the model. "
    "Drayage and Long Haul CNG currently use diesel-like trial parameters so you can test the workflow before replacing them with final CNG data."
)
