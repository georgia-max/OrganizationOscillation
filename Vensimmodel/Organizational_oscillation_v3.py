"""
Python model 'Organizational_oscillation_v3.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.functions import step, ramp, pulse
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
    "final_time": lambda: 50,
    "time_step": lambda: 0.5,
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
    name="FINAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Year",
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
    units="Year",
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
    name="Change in Safety Focus",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "effect_of_nonsafety_performance_on_chge_sf": 1,
        "time_to_adjust_focus": 1,
    },
)
def change_in_safety_focus():
    return effect_of_nonsafety_performance_on_chge_sf() / time_to_adjust_focus()


@component.add(
    name='"Change in Non-Safety Focus"',
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"safety_performance_gap": 1, "time_to_adjust_focus": 1},
)
def change_in_nonsafety_focus():
    return safety_performance_gap() / time_to_adjust_focus()


@component.add(
    name='"Effect of Non-Safety Performance on Chge SF"',
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"nonsafety_performance": 1},
)
def effect_of_nonsafety_performance_on_chge_sf():
    return nonsafety_performance()


@component.add(
    name="Safety Focus",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_safety_focus": 1},
    other_deps={
        "_integ_safety_focus": {"initial": {}, "step": {"change_in_safety_focus": 1}}
    },
)
def safety_focus():
    return _integ_safety_focus()


_integ_safety_focus = Integ(
    lambda: change_in_safety_focus(), lambda: 0, "_integ_safety_focus"
)


@component.add(
    name='"Non-Safety focus"',
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_nonsafety_focus": 1},
    other_deps={
        "_integ_nonsafety_focus": {
            "initial": {},
            "step": {"change_in_nonsafety_focus": 1},
        }
    },
)
def nonsafety_focus():
    return _integ_nonsafety_focus()


_integ_nonsafety_focus = Integ(
    lambda: change_in_nonsafety_focus(), lambda: 1, "_integ_nonsafety_focus"
)


@component.add(
    name="Safety Performance Gap",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"serious_errors": 1, "safety_focus": 1},
)
def safety_performance_gap():
    return serious_errors() - safety_focus()


@component.add(name="Pulse Quantity 1", comp_type="Constant", comp_subtype="Normal")
def pulse_quantity_1():
    return 0


@component.add(name="Pulse repeat time", comp_type="Constant", comp_subtype="Normal")
def pulse_repeat_time():
    return 0


@component.add(name="Pulse end time", comp_type="Constant", comp_subtype="Normal")
def pulse_end_time():
    return 25


@component.add(name="Pulse start time", comp_type="Constant", comp_subtype="Normal")
def pulse_start_time():
    return 1


@component.add(
    name="input",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "step_height": 1,
        "step_time": 1,
        "time": 4,
        "time_step": 2,
        "pulse_quantity_1": 1,
        "pulse_time": 1,
        "pulse_duration": 2,
        "ramp_start_time": 1,
        "ramp_end_time": 1,
        "ramp_slope": 1,
        "pulse_repeat_time": 1,
        "pulse_start_time": 1,
        "pulse_end_time": 1,
        "pulse_quantity": 1,
    },
)
def input_1():
    return (
        step(__data["time"], step_height(), step_time())
        + (pulse_quantity_1() / time_step())
        * pulse(__data["time"], pulse_time(), width=pulse_duration())
        + ramp(__data["time"], ramp_slope(), ramp_start_time(), ramp_end_time())
        + pulse(
            __data["time"],
            pulse_start_time(),
            repeat_time=pulse_repeat_time(),
            width=pulse_duration(),
            end=pulse_end_time(),
        )
        * (pulse_quantity() / time_step())
    )


@component.add(name="Pulse Duration", comp_type="Constant", comp_subtype="Normal")
def pulse_duration():
    return 0


@component.add(
    name="Serious Errors",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"initial_safety_threshold": 1, "input_1": 1},
)
def serious_errors():
    return initial_safety_threshold() + input_1()


@component.add(name="Ramp End Time", comp_type="Constant", comp_subtype="Normal")
def ramp_end_time():
    return 0


@component.add(name="Ramp Slope", comp_type="Constant", comp_subtype="Normal")
def ramp_slope():
    return 0


@component.add(name="Ramp Start Time", comp_type="Constant", comp_subtype="Normal")
def ramp_start_time():
    return 0


@component.add(
    name="initial Safety Threshold", comp_type="Constant", comp_subtype="Normal"
)
def initial_safety_threshold():
    return 0


@component.add(name="Step Height", comp_type="Constant", comp_subtype="Normal")
def step_height():
    return 0


@component.add(name="Pulse Quantity", comp_type="Constant", comp_subtype="Normal")
def pulse_quantity():
    return 0


@component.add(name="Pulse Time", comp_type="Constant", comp_subtype="Normal")
def pulse_time():
    return 0


@component.add(name="Step Time", comp_type="Constant", comp_subtype="Normal")
def step_time():
    return 0


@component.add(
    name="Time to adjust focus",
    units="Month",
    limits=(0.0, 20.0, 2.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_to_adjust_focus():
    return 2


@component.add(
    name="Normal NonSafety performance", comp_type="Constant", comp_subtype="Normal"
)
def normal_nonsafety_performance():
    return 0


@component.add(
    name='"Non-Safety Performance"',
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"nonsafety_focus": 1, "normal_nonsafety_performance": 1},
)
def nonsafety_performance():
    return nonsafety_focus() - normal_nonsafety_performance()
