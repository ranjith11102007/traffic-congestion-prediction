"""Backend package for the Traffic Congestion Prediction API.

Sub-packages are intentionally left out of this module: importing ``app`` must
stay cheap and free of side effects so tooling, tests and the ML pipeline never
drag in the HTTP layer.
"""

__version__ = "0.1.0"