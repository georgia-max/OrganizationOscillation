"""
Python model 'model_13.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np
import xarray as xr
from scipy import stats

from pysd.py_backend.functions import xidz, sum, if_then_else
from pysd.py_backend.statefuls import Integ, Smooth, Initial
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


_subscript_dict = {"Goal": ["A", "B"]}

component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 100,
    "time_step": lambda: 0.0625,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="performance",
    units="Dmnl",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "caa": 1,
        "resources": 4,
        "cba": 1,
        "sw_a_to_protective": 2,
        "resource_generative_outcome": 2,
        "shock_effect_on_performance": 2,
        "cab": 1,
        "cbb": 1,
        "sw_b_to_protective": 2,
    },
)
def performance():
    """
    add stochasticity potentially generative vs protective generative can be additive with a small stochastic component and for protective with can have a rare event large impact (like safety) combined performance determines endogenous resource inflow of future
    """
    value = xr.DataArray(np.nan, {"Goal": _subscript_dict["Goal"]}, ["Goal"])
    value.loc[["A"]] = (
        caa() * float(resources().loc["A"])
        + cba() * float(resources().loc["B"])
        + (1 - sw_a_to_protective()) * resource_generative_outcome()
        + sw_a_to_protective() * shock_effect_on_performance()
    )
    value.loc[["B"]] = (
        cab() * float(resources().loc["A"])
        + cbb() * float(resources().loc["B"])
        + (1 - sw_b_to_protective()) * resource_generative_outcome()
        + sw_b_to_protective() * shock_effect_on_performance()
    )
    return value


@component.add(
    name="sw B to protective",
    limits=(0.0, 1.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def sw_b_to_protective():
    return 1


@component.add(
    name="change in asp",
    units="1/Month",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "perceived_perfromance_gap": 1,
        "sw_endogen_aspiration": 1,
        "k_asp": 1,
        "accident_shock_level": 1,
        "t_change_asp": 1,
    },
)
def change_in_asp():
    """
    there should be some nonlinearity here, if perf gap is zero, change in aspirations should be positive rather than zero. Georgia: adding (1+k asp*Accident shock level), with big accidents, aspiration accelerates. -> crisis reveal aspirations. “Aspirations adjust to the performance gap at a base rate determined by t change asp, but this rate is multiplied by (1 + sensitivity × shock). If shocks are present, managers shorten the time horizon for adjustment — they react faster.”
    """
    return (
        -perceived_perfromance_gap()
        * sw_endogen_aspiration()
        * (1 + k_asp() * accident_shock_level())
        / t_change_asp()
    )


@component.add(
    name="accident rate", units="1/Month", comp_type="Constant", comp_subtype="Normal"
)
def accident_rate():
    return 1


@component.add(name="accident severity", comp_type="Constant", comp_subtype="Normal")
def accident_severity():
    """
    Magnitude of performance loss per accident event
    """
    return 10


@component.add(
    name="Accident shock level",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_accident_shock_level": 1},
    other_deps={
        "_integ_accident_shock_level": {
            "initial": {},
            "step": {"accidents": 1, "recovery_rate": 1},
        }
    },
)
def accident_shock_level():
    return _integ_accident_shock_level()


_integ_accident_shock_level = Integ(
    lambda: accidents() - recovery_rate(), lambda: 0, "_integ_accident_shock_level"
)


@component.add(
    name="accidents",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "accident_rate": 1,
        "time_step": 1,
        "seed": 1,
        "time": 1,
        "accident_severity": 1,
    },
)
def accidents():
    return (
        -float(
            np.clip(
                np.random.poisson(lam=accident_rate() * time_step(), size=()) * 1 + 0,
                0,
                5,
            )
        )
        * accident_severity()
    )


@component.add(
    name="recovery time",
    units="Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"recovery_time_base": 1, "k_asp": 1, "total_investments": 1},
)
def recovery_time():
    return float(
        np.maximum(0.1, recovery_time_base() / (1 + k_asp() * total_investments()))
    )


@component.add(
    name="total investments",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"resources": 1},
)
def total_investments():
    return sum(resources().rename({"Goal": "Goal!"}), dim=["Goal!"])


@component.add(
    name="recovery rate",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"accident_shock_level": 1, "recovery_time": 1},
)
def recovery_rate():
    return accident_shock_level() / recovery_time()


@component.add(
    name="k asp", limits=(0.001, 0.5), comp_type="Constant", comp_subtype="Normal"
)
def k_asp():
    """
    Sensitivity of aspiration updating to shocks. Higher = aspirations move faster when accidents occur
    """
    return 0.02


@component.add(
    name="recovery time base",
    units="Month",
    limits=(5.0, 30.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def recovery_time_base():
    return 12


@component.add(
    name="shock sensitivity", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def shock_sensitivity():
    return 1


@component.add(
    name="shock effect on performance",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"accident_shock_level": 1, "shock_sensitivity": 1},
)
def shock_effect_on_performance():
    return accident_shock_level() * shock_sensitivity()


@component.add(
    name="sw A to protective",
    limits=(0.0, 1.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def sw_a_to_protective():
    return 1


@component.add(
    name="resource generative outcome",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gstdv": 1, "seed": 1, "time": 1},
)
def resource_generative_outcome():
    return float(
        stats.truncnorm.rvs(
            xidz(-10 - 0, gstdv(), -np.inf),
            xidz(10 - 0, gstdv(), np.inf),
            loc=0,
            scale=gstdv(),
            size=(),
        )
    )


@component.add(
    name="aspiration",
    units="Dmnl",
    limits=(0.0, 1000.0, 1.0),
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_aspiration": 1},
    other_deps={
        "_integ_aspiration": {
            "initial": {"exogen_aspiration": 1},
            "step": {"change_in_asp": 1},
        }
    },
)
def aspiration():
    """
    if perfromance is consistently falling short, adjust aspiration down if aspirations are consistently met, adjust them up future: market trends, ecosystem dynamics
    """
    return _integ_aspiration()


_integ_aspiration = Integ(
    lambda: change_in_asp(),
    lambda: xr.DataArray(
        exogen_aspiration(), {"Goal": _subscript_dict["Goal"]}, ["Goal"]
    ),
    "_integ_aspiration",
)


@component.add(
    name="init perc comb perf",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Initial",
    depends_on={"_initial_init_perc_comb_perf": 1},
    other_deps={
        "_initial_init_perc_comb_perf": {
            "initial": {
                "caa": 1,
                "init_a": 2,
                "cba": 1,
                "init_b": 2,
                "cab": 1,
                "cbb": 1,
            },
            "step": {},
        }
    },
)
def init_perc_comb_perf():
    return _initial_init_perc_comb_perf()


_initial_init_perc_comb_perf = Initial(
    lambda: caa() * init_a() + cba() * init_b() + cab() * init_a() + cbb() * init_b(),
    "_initial_init_perc_comb_perf",
)


@component.add(
    name="Perceived comb perf",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_perceived_comb_perf": 1},
    other_deps={
        "_integ_perceived_comb_perf": {
            "initial": {"init_perc_comb_perf": 1},
            "step": {"change_in_perc_comb_perf": 1},
        }
    },
)
def perceived_comb_perf():
    return _integ_perceived_comb_perf()


_integ_perceived_comb_perf = Integ(
    lambda: change_in_perc_comb_perf(),
    lambda: init_perc_comb_perf(),
    "_integ_perceived_comb_perf",
)


@component.add(
    name="Perceived perfromance gap",
    units="Dmnl",
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_perceived_perfromance_gap": 1},
    other_deps={
        "_smooth_perceived_perfromance_gap": {
            "initial": {"performance_gap": 1},
            "step": {"performance_gap": 1, "t_update_perf_gap": 1},
        }
    },
)
def perceived_perfromance_gap():
    return _smooth_perceived_perfromance_gap()


_smooth_perceived_perfromance_gap = Smooth(
    lambda: performance_gap(),
    lambda: xr.DataArray(
        t_update_perf_gap(), {"Goal": _subscript_dict["Goal"]}, ["Goal"]
    ),
    lambda: performance_gap(),
    lambda: 1,
    "_smooth_perceived_perfromance_gap",
)


@component.add(
    name="change in perc comb perf",
    units="1/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "combined_performance": 2,
        "perceived_comb_perf": 2,
        "t_adj_perc_downwards": 1,
        "t_adj_perc_upwards": 1,
    },
)
def change_in_perc_comb_perf():
    return (combined_performance() - perceived_comb_perf()) / if_then_else(
        combined_performance() > perceived_comb_perf(),
        lambda: t_adj_perc_upwards(),
        lambda: t_adj_perc_downwards(),
    )


@component.add(
    name="t adj perc upwards",
    units="Month",
    limits=(0.01, 50.0, 0.5),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_adj_perc_upwards():
    return 20


@component.add(
    name="resource inflow",
    units="Dollar/Month",
    limits=(0.0, 100.0, 0.1),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"sw_endog_inflow": 2, "endogen_resource_inflow": 1, "exogen_inflow": 1},
)
def resource_inflow():
    """
    endogenous inflow: function of performance (combined performance)
    """
    return (
        sw_endog_inflow() * endogen_resource_inflow()
        + (1 - sw_endog_inflow()) * exogen_inflow()
    )


@component.add(
    name="endogen resource inflow",
    units="Dollar/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"sens_res_inflow": 2, "inflow_per_perf": 1, "perceived_comb_perf": 1},
)
def endogen_resource_inflow():
    """
    https://www.desmos.com/calculator/dnhbwdyqdq
    """
    return (1 / sens_res_inflow()) * float(
        np.log(
            1
            + float(
                np.exp(sens_res_inflow() * inflow_per_perf() * perceived_comb_perf())
            )
        )
    )


@component.add(
    name="seed", limits=(1.0, 5000.0, 1.0), comp_type="Constant", comp_subtype="Normal"
)
def seed():
    return 1


@component.add(
    name="exogen aspiration",
    units="Dmnl",
    limits=(0.0, 1000.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def exogen_aspiration():
    """
    if perfromance is consistently falling short, adjust aspiration down if aspirations are consistently met, adjust them up future: market trends, ecosystem dynamics
    """
    return 100


@component.add(
    name="sens res inflow",
    units="Dmnl",
    limits=(0.0, 2.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def sens_res_inflow():
    return 1


@component.add(
    name="t change asp",
    units="Month",
    limits=(1.0, 50.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_change_asp():
    return 40


@component.add(
    name="gstdv",
    units="Dmnl",
    limits=(0.0, 30.0, 0.5),
    comp_type="Constant",
    comp_subtype="Normal",
)
def gstdv():
    return 0


@component.add(
    name="inflow per perf",
    units="Dollar/Month",
    limits=(0.0, 0.5, 0.001),
    comp_type="Constant",
    comp_subtype="Normal",
)
def inflow_per_perf():
    return 0.05


@component.add(
    name="t adj perc downwards",
    units="Month",
    limits=(0.01, 50.0, 0.5),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_adj_perc_downwards():
    return 2


@component.add(
    name="t update perf gap",
    units="Month",
    limits=(1.0, 20.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_update_perf_gap():
    return 5


@component.add(
    name="performance gap",
    units="Dmnl",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"aspiration": 1, "performance": 1},
)
def performance_gap():
    return aspiration() - performance()


@component.add(
    name="sw endog inflow",
    units="Dmnl",
    limits=(0.0, 1.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def sw_endog_inflow():
    return 0


@component.add(
    name="sw endogen aspiration",
    units="Dmnl",
    limits=(0.0, 1.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def sw_endogen_aspiration():
    return 0


@component.add(
    name="init resource",
    units="Dollar",
    limits=(0.0, 200.0, 1.0),
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"init_a": 1, "init_b": 1},
)
def init_resource():
    value = xr.DataArray(np.nan, {"Goal": _subscript_dict["Goal"]}, ["Goal"])
    value.loc[["A"]] = init_a()
    value.loc[["B"]] = init_b()
    return value


@component.add(
    name="combined performance",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"performance": 1},
)
def combined_performance():
    return sum(performance().rename({"Goal": "Goal!"}), dim=["Goal!"])


@component.add(
    name="init res",
    units="Dollar",
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Initial",
    depends_on={"_initial_init_res": 1},
    other_deps={
        "_initial_init_res": {"initial": {"eq_init": 1, "init_resource": 1}, "step": {}}
    },
)
def init_res():
    return _initial_init_res()


_initial_init_res = Initial(
    lambda: (1 - eq_init()) * init_resource(), "_initial_init_res"
)


@component.add(
    name="inv weight",
    units="Dmnl",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"softmax_snstvty": 1, "performance_gap": 1, "pg_denum": 1},
)
def inv_weight():
    return np.exp(softmax_snstvty() * performance_gap()) / pg_denum()


@component.add(
    name="deterioration resources",
    units="Dollar/Month",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"resources": 1, "t_deteriorate": 1},
)
def deterioration_resources():
    return resources() / t_deteriorate()


@component.add(
    name="Resources",
    units="Dollar",
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_resources": 1},
    other_deps={
        "_integ_resources": {
            "initial": {"init_res": 1},
            "step": {"maturing_inv": 1, "deterioration_resources": 1},
        }
    },
)
def resources():
    return _integ_resources()


_integ_resources = Integ(
    lambda: maturing_inv() - deterioration_resources(),
    lambda: init_res(),
    "_integ_resources",
)


@component.add(
    name="init A",
    units="Dollar",
    limits=(0.0, 200.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def init_a():
    return 80


@component.add(
    name="pg denum",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"softmax_snstvty": 1, "performance_gap": 1},
)
def pg_denum():
    """
    (exp(softmax snstvty * performance gap[A]) + exp(softmax snstvty * performance gap[B]))
    """
    return sum(
        np.exp(softmax_snstvty() * performance_gap().rename({"Goal": "Goal!"})),
        dim=["Goal!"],
    )


@component.add(
    name="init B",
    units="Dollar",
    limits=(0.0, 200.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def init_b():
    return 80


@component.add(
    name="maturing inv",
    units="Dollar/Month",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"undeveloped_resources": 1, "t_mature": 1},
)
def maturing_inv():
    return undeveloped_resources() / t_mature()


@component.add(
    name="init UR",
    units="Dollar",
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Initial",
    depends_on={"_initial_init_ur": 1},
    other_deps={
        "_initial_init_ur": {
            "initial": {"init_res": 1, "t_deteriorate": 1, "t_mature": 1},
            "step": {},
        }
    },
)
def init_ur():
    return _initial_init_ur()


_initial_init_ur = Initial(
    lambda: (init_res() / t_deteriorate()) * t_mature(), "_initial_init_ur"
)


@component.add(
    name="t mature",
    units="Month",
    limits=(1.0, 20.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_mature():
    return 5


@component.add(
    name="Undeveloped Resources",
    units="Dollar",
    subscripts=["Goal"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_undeveloped_resources": 1},
    other_deps={
        "_integ_undeveloped_resources": {
            "initial": {"init_ur": 1},
            "step": {"investing": 1, "maturing_inv": 1},
        }
    },
)
def undeveloped_resources():
    return _integ_undeveloped_resources()


_integ_undeveloped_resources = Integ(
    lambda: investing() - maturing_inv(),
    lambda: init_ur(),
    "_integ_undeveloped_resources",
)


@component.add(
    name="exogen inflow",
    units="Dollar/Month",
    limits=(0.0, 20.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def exogen_inflow():
    return 8


@component.add(
    name="eq init",
    units="Dmnl",
    limits=(0.0, 1.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def eq_init():
    return 0


@component.add(
    name="softmax snstvty",
    units="Dmnl",
    limits=(0.0, 0.5, 0.001),
    comp_type="Constant",
    comp_subtype="Normal",
)
def softmax_snstvty():
    return 0.03


@component.add(
    name="investing",
    units="Dollar/Month",
    subscripts=["Goal"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"inv_weight": 1, "resource_inflow": 1},
)
def investing():
    return inv_weight() * resource_inflow()


@component.add(
    name="cAB",
    units="1/Dollar",
    limits=(-1.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def cab():
    return 0


@component.add(
    name="cBA",
    units="1/Dollar",
    limits=(-1.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def cba():
    return 0


@component.add(
    name="cAA",
    units="1/Dollar",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def caa():
    return 1


@component.add(
    name="cBB",
    units="1/Dollar",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def cbb():
    return 1


@component.add(
    name="t deteriorate",
    units="Month",
    limits=(1.0, 50.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_deteriorate():
    return 25
