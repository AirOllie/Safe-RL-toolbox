"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys


# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.simzoo.envs.classic_control.quad3d.quad3d_normal import Quad3D
elif importlib.util.find_spec("simzoo") is not None:
    Quad3D = getattr(
        importlib.import_module("simzoo.envs.classic_control.quad3d.quad3d"),
        "Quad3D",
    )
else:
    Quad3D = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.quad3d.quad3d"
        ),
        "Quad3D",
    )
