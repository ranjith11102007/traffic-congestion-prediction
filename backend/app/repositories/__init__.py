"""Data-access layer.

Every SQL statement in the application lives in a repository module, which is why
route handlers never import SQLAlchemy query helpers. Repositories receive a
:class:`~sqlalchemy.orm.Session` rather than opening one, so the caller owns the
transaction boundary and a test can share a single connection across cases.

Not every function is re-exported here. This module deliberately stays thin: the
submodules are imported as ``from app.repositories import traffic_repository`` so a
caller is explicit about which table it is touching, and a mis-named import fails
at the point of use instead of resolving to an identically named function.
"""

__all__: list[str] = []