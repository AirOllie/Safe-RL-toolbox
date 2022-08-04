"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys


# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.simzoo.envs.classic_control.quad2d.quad2d_nomal import Quad2D
elif importlib.util.find_spec("simzoo") is not None:
    Quad2D = getattr(
        importlib.import_module("simzoo.envs.classic_control.quad2d.quad2d"),
        "Quad2D",
    )
else:
    Quad2D = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.quad2d.quad2d"
        ),
        "Quad2D",
    )
