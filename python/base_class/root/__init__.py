"""
High-level tools for ROOT file I/O built on top of :mod:`uproot`.

.. note::
    :mod:`pandas` will not be imported unless necessary.
"""
from .chunk import Chunk
from .io import TreeReader, TreeWriter

__all__ = [
    'Chunk',
    'TreeReader',
    'TreeWriter',
]
