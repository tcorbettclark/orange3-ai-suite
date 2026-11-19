"""Images as table rows: a row of luma values in [0, 1], and the features described
from each image.

:mod:`decode` reads the caller's array, :mod:`resize` scales it to a square,
:mod:`flatten` names each cell, and :mod:`pixels` composes those three steps.
``features`` describes the same image through the same decode and resize.
"""
