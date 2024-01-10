"""
Python model 'pinknoise_test.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.9.1"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 50,
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
    name="updating pink noise",
    units="widgets/week/week",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"white_noise": 1, "auto_noise": 1, "correlation_time": 1},
)
def updating_pink_noise():
    return (white_noise() - auto_noise()) / correlation_time()


@component.add(
    name="pink noise",
    units="widgets/week",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"auto_noise": 1},
)
def pink_noise():
    return auto_noise()


@component.add(
    name="auto noise",
    units="widgets/week",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_auto_noise": 1},
    other_deps={
        "_integ_auto_noise": {"initial": {}, "step": {"updating_pink_noise": 1}}
    },
)
def auto_noise():
    """
    Contributed by Ed Anderson, MIT/U. Texas - Austin Description: The pink noise molecule described generates a simple random series with autocorrelation. This is useful in representing time series, like rainfall from day to day, in which today's value has some correlation with what happened yesterday. This particular formulation will also have properties such as standard deviation and mean that are insensitive both to the time step and the correlation (smoothing) time. Finally, the output as a whole and the difference in values between any two days is guaranteed to be Gaussian (normal) in distribution. Behavior: Pink noise series will have both a historical and a random component during each period. The relative "trend-to-noise" ratio is controlled by the length of the correlation time. As the correlation time approaches zero, the pink noise output will become more independent of its historical value and more "noisy." On the other hand, as the correlation time approaches infinity, the pink noise output will approximate a continuous time random walk or Brownian motion. Displayed above are two time series with correlation times of 1 and 8 months. While both series have approximately the same standard deviation, the 1-month correlation time series is less smooth from period to period than the 8-month series, which is characterized by "sustained" swings in a given direction. Note that this behavior will be independent of the time-step. The "pink" in pink noise refers to the power spectrum of the output. A time series in which each period's observation is independent of the past is characterized by a flat or "white" power spectrum. Smoothing a time series attenuates the higher or "bluer" frequencies of the power spectrum, leaving the lower or "redder" frequencies relatively stronger in the output. Caveats: This assumes the use of Euler integration with a time step of no more than 1/4 of the correlation time. Very long correlation times should be avoided also as the multiplication in the scaled white noise will become progressively less accurate. Technical Notes: This particular form of pink noise is superior to that of Britting presented in Richardson and Pugh (1981) because the Gaussian (Normal) distribution of the output does not depend on the Central Limit Theorem. (Dynamo did not have a Gaussian random number generator and hence R&P had to invoke the CLM to get a normal distribution.) Rather, this molecule's normal output is a result of the observations being a sum of Gaussian draws. Hence, the series over short intervals should better approximate normality than the macro in R&P. MEAN: This is the desired mean for the pink noise. STD DEVIATION: This is the desired standard deviation for the pink noise. CORRELATION TIME: This is the smooth time for the noise, or for the more technically minded this is the inverse of the filter's cut-off frequency in radians. Updated by Tom Fiddaman, 2010, to include a random initial value, correct units, and use TIME STEP$ keyword
    """
    return _integ_auto_noise()


_integ_auto_noise = Integ(lambda: updating_pink_noise(), lambda: 0, "_integ_auto_noise")


@component.add(
    name="white noise",
    units="widgets/week",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "mean": 1,
        "std_deviation": 1,
        "correlation_time": 1,
        "time": 1,
        "time_step": 1,
        "seed": 1,
    },
)
def white_noise():
    """
    This adjusts the standard deviation of the white noise to compensate for the time step and the correlation time to produce the appropriate pink noise std deviation.
    """
    return mean() + std_deviation() * (
        24 * correlation_time() / time_step()
    ) ** 0.5 * np.random.uniform(-0.5, 0.5, size=())


@component.add(
    name="seed",
    units="dmnl",
    limits=(1.0, 10000.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def seed():
    return 1


@component.add(
    name="correlation time",
    units="week",
    limits=(1.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def correlation_time():
    return 10


@component.add(
    name="mean",
    units="widgets/week",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def mean():
    return 0


@component.add(
    name="std deviation",
    units="widgets/week",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def std_deviation():
    return 10
