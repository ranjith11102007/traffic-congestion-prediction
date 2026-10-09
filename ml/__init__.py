"""Machine-learning package for the traffic congestion prediction system.

This package is deliberately **independent of the FastAPI application**: it never
imports ``app`` and imports no web framework. That separation lets the model be
trained, evaluated and benchmarked on a machine with no API running, and lets the
API be deployed without the ML stack installed.

Module layout
-------------
``preprocessing``  dataset loading, validation, feature engineering, target derivation
``train``          estimator construction, training loop, artifact IO
``evaluate``       metrics for a trained artifact
``predict``        inference against a trained artifact

Status: Phase 1 foundation. The code paths are implemented and importable, but
no model artifact exists yet because no dataset has been acquired. Every entry
point fails loudly rather than producing fabricated numbers.
"""

__version__ = "0.1.0"