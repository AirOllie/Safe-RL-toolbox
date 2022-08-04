"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys


# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.simzoo.envs.classic_control.quad_crazyflie.quad_crazyflie import (
        Crazyflie,
    )
elif importlib.util.find_spec("simzoo") is not None:
    Crazyflie = getattr(
        importlib.import_module(
            "simzoo.envs.classic_control.quad_crazyflie.quad_crazyflie"
        ),
        "Crazyflie",
    )
else:
    Crazyflie = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.quad_crazyflie.quad_crazyflie"
        ),
        "Crazyflie",
    )
