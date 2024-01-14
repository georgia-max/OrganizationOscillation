"""
Python model 'safety focus_v11(Noise).py'
Translated using PySD
"""

from pathlib import Path
import numpy as np
from scipy import stats

from pysd.py_backend.functions import if_then_else, integer
from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.9.0"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


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
    name="Error Occurance",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 2, "pink_noise": 1, "chance_of_errors": 1},
)
def error_occurance():
    """
    IF THEN ELSE( INTEGER(Time)=Time, IF THEN ELSE(RANDOM UNIFORM(0, 1, seed)<Chance of Errors, 1, 0),0) IF THEN ELSE( SWTICH = 0 , IF THEN ELSE( INTEGER(Time)=Time, IF THEN ELSE(random norm<Chance of Errors, 1, 0),0) , exogenous shock )
    """
    return if_then_else(
        integer(time()) == time(),
        lambda: if_then_else(pink_noise() < chance_of_errors(), lambda: 1, lambda: 0),
        lambda: 0,
    )


@component.add(name="mean", comp_type="Constant", comp_subtype="Normal")
def mean():
    return 0.15


@component.add(name="seed2", comp_type="Constant", comp_subtype="Normal")
def seed2():
    return 1


@component.add(name="std", comp_type="Constant", comp_subtype="Normal")
def std():
    return 0.2


@component.add(
    name="white noise",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"mean": 1, "time_step": 1, "seed2": 1, "std": 1, "correlation_time": 1},
)
def white_noise():
    return mean() + std() * (
        24 * correlation_time() / time_step()
    ) ** 0.5 * np.random.uniform(-0.5, 0.5, size=())


@component.add(
    name="correlation time",
    limits=(1.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def correlation_time():
    return 10


@component.add(
    name="Auto noise",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_auto_noise": 1},
    other_deps={
        "_integ_auto_noise": {"initial": {}, "step": {"updating_pink_noise": 1}}
    },
)
def auto_noise():
    return _integ_auto_noise()


_integ_auto_noise = Integ(lambda: updating_pink_noise(), lambda: 0, "_integ_auto_noise")


@component.add(
    name="updating pink noise",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"white_noise": 1, "auto_noise": 1, "correlation_time": 1},
)
def updating_pink_noise():
    return (white_noise() - auto_noise()) / correlation_time()


@component.add(
    name="pink noise",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"auto_noise": 1},
)
def pink_noise():
    return auto_noise()


@component.add(
    name="Implied safety performance",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"organization_knowledge": 1},
)
def implied_safety_performance():
    return organization_knowledge()


@component.add(
    name="Chance of Errors",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "gamma": 1,
        "safety_threshold": 1,
        "implied_safety_performance": 1,
        "pressure_from_manager": 1,
    },
)
def chance_of_errors():
    return gamma() * (
        1
        - 1
        / (
            1
            + np.exp(
                -pressure_from_manager()
                * (implied_safety_performance() - safety_threshold())
            )
        )
    )


@component.add(
    name="Attention on Safety",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "attention_capability": 1,
        "normal_attention_on_safety": 1,
        "attention_resource": 1,
    },
)
def attention_on_safety():
    return 1 / (
        1
        + np.exp(
            -attention_capability()
            * (attention_resource() - normal_attention_on_safety())
        )
    )


@component.add(name="gamma", comp_type="Constant", comp_subtype="Normal")
def gamma():
    return 0.1


@component.add(
    name="depreciation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "organization_knowledge": 1,
        "attention_on_safety": 1,
        "t_to_depreciate": 1,
    },
)
def depreciation():
    return (
        np.maximum(0, organization_knowledge() - attention_on_safety())
        / t_to_depreciate()
    )


@component.add(
    name="Organization knowledge",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_organization_knowledge": 1},
    other_deps={
        "_integ_organization_knowledge": {
            "initial": {"initial_state": 1},
            "step": {"experience": 1, "depreciation": 1},
        }
    },
)
def organization_knowledge():
    return _integ_organization_knowledge()


_integ_organization_knowledge = Integ(
    lambda: experience() - depreciation(),
    lambda: initial_state(),
    "_integ_organization_knowledge",
)


@component.add(
    name="Attention growth",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "desired_attention_goal": 1,
        "attention_resource": 1,
        "error_occurance": 1,
        "t_to_grow_attention": 1,
    },
)
def attention_growth():
    return np.maximum(
        0, (desired_attention_goal() - attention_resource()) * error_occurance()
    ) / (t_to_grow_attention() / 16)


@component.add(
    name="T to depreciate",
    limits=(1.0, 50.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_to_depreciate():
    return 5


@component.add(
    name="experience",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "attention_on_safety": 1,
        "organization_knowledge": 1,
        "t_to_gain_experience": 1,
    },
)
def experience():
    return np.maximum(
        0, (attention_on_safety() - organization_knowledge()) / t_to_gain_experience()
    )


@component.add(
    name="T to grow attention",
    limits=(1.0, 5.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_to_grow_attention():
    return 1


@component.add(
    name="Normal Attention on Safety",
    limits=(0.0, 1.0, 0.02),
    comp_type="Constant",
    comp_subtype="Normal",
)
def normal_attention_on_safety():
    return 0.5


@component.add(
    name="T to gain experience",
    limits=(1.0, 50.0, 0.5),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_to_gain_experience():
    return 10


@component.add(
    name="attention capability",
    limits=(0.5, 30.0, 0.2),
    comp_type="Constant",
    comp_subtype="Normal",
)
def attention_capability():
    return 10


@component.add(
    name="desired attention goal",
    limits=(0.5, 1.0, 0.02),
    comp_type="Constant",
    comp_subtype="Normal",
)
def desired_attention_goal():
    return 1


@component.add(
    name="initial focus",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"seed": 1},
)
def initial_focus():
    return stats.truncnorm.rvs(0, 1, loc=0.5, scale=0.2, size=())


@component.add(
    name="initial state",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"seed": 1},
)
def initial_state():
    return stats.truncnorm.rvs(0, 1, loc=0.5, scale=0.2, size=())


@component.add(
    name="pressure from manager",
    limits=(0.0, 20.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def pressure_from_manager():
    return 10


@component.add(
    name="Attention Resource",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_attention_resource": 1},
    other_deps={
        "_integ_attention_resource": {
            "initial": {"initial_focus": 1},
            "step": {"attention_growth": 1, "attention_erosion": 1},
        }
    },
)
def attention_resource():
    return _integ_attention_resource()


_integ_attention_resource = Integ(
    lambda: attention_growth() - attention_erosion(),
    lambda: initial_focus(),
    "_integ_attention_resource",
)


@component.add(
    name="Safety threshold",
    limits=(0.0, 1.0, 0.02),
    comp_type="Constant",
    comp_subtype="Normal",
)
def safety_threshold():
    return 0.5


@component.add(
    name="seed", limits=(0.0, 100.0, 1.0), comp_type="Constant", comp_subtype="Normal"
)
def seed():
    return 1


@component.add(
    name="Attention erosion",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"attention_resource": 1, "t_to_lose_attention": 1},
)
def attention_erosion():
    return attention_resource() / t_to_lose_attention()


@component.add(
    name="T to lose attention",
    limits=(1.0, 50.0, 0.5),
    comp_type="Constant",
    comp_subtype="Normal",
)
def t_to_lose_attention():
    return 10