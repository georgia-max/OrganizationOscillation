"""
Python model 'Organizational_oscillation_v4.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.functions import if_then_else, step, ramp, pulse
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
    name="Acceptable Safety Performance",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def acceptable_safety_performance():
    return 5


@component.add(
    name="Accident",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1},
)
def accident():
    return pulse(__data["time"], 5, repeat_time=3, width=1, end=20) * 5


@component.add(
    name="Error Margin",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"acceptable_safety_performance": 1, "safety_focus": 1},
)
def error_margin():
    return acceptable_safety_performance() - safety_focus()


@component.add(
    name="Serious Error",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"constant": 1, "function": 1},
)
def serious_error():
    return constant() + function()


@component.add(
    name="Pressure to focus on Safety",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "nonsafety_performance": 1,
        "switch": 1,
        "serious_error": 1,
        "accident": 1,
    },
)
def pressure_to_focus_on_safety():
    return nonsafety_performance() + if_then_else(
        switch() == 0, lambda: serious_error(), lambda: accident()
    )


@component.add(
    name="SWITCH", limits=(0.0, 1.0, 1.0), comp_type="Constant", comp_subtype="Normal"
)
def switch():
    return 1


@component.add(
    name="Change in Safety Focus",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"pressure_to_focus_on_safety": 1, "time_to_adjust_focus": 1},
)
def change_in_safety_focus():
    return pressure_to_focus_on_safety() / time_to_adjust_focus()


@component.add(
    name="Change in NonSafety Focus",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"error_margin": 1, "time_to_adjust_focus": 1},
)
def change_in_nonsafety_focus():
    return error_margin() / time_to_adjust_focus()


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
    name="NonSafety Focus",
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


@component.add(name="Pulse Quantity 1", comp_type="Constant", comp_subtype="Normal")
def pulse_quantity_1():
    return 1


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
    name="function",
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
def function():
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
    return 1


@component.add(name="Ramp End Time", comp_type="Constant", comp_subtype="Normal")
def ramp_end_time():
    return 0


@component.add(name="Ramp Slope", comp_type="Constant", comp_subtype="Normal")
def ramp_slope():
    return 0


@component.add(name="Ramp Start Time", comp_type="Constant", comp_subtype="Normal")
def ramp_start_time():
    return 0


@component.add(name="Constant", comp_type="Constant", comp_subtype="Normal")
def constant():
    return 0


@component.add(name="Step Height", comp_type="Constant", comp_subtype="Normal")
def step_height():
    return 0


@component.add(name="Pulse Quantity", comp_type="Constant", comp_subtype="Normal")
def pulse_quantity():
    return 1


@component.add(name="Pulse Time", comp_type="Constant", comp_subtype="Normal")
def pulse_time():
    return 5


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
    return 5


@component.add(
    name="Normal NonSafety performance", comp_type="Constant", comp_subtype="Normal"
)
def normal_nonsafety_performance():
    """
    PULSE( 5 , 1 )
    """
    return 0


@component.add(
    name="NonSafety Performance",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"nonsafety_focus": 1, "normal_nonsafety_performance": 1},
)
def nonsafety_performance():
    return nonsafety_focus() - normal_nonsafety_performance()
