"""Image data: what the caller handed us, and what can be measured from it.

Five parts, each with a reason to change of its own:

- :mod:`decode` - what the caller handed us: alpha, channel order, value
  range. Parameterless, and it refuses a range it cannot interpret.
- :mod:`resize` - resampling to a fixed square, one policy per axis: area
  averaging when an axis shrinks, bilinear when it grows.
- :mod:`flatten` - a 2-D grid as one row, with a coordinate name per cell:
  the columns of a pixel-values table.
- :mod:`pixels` - the plain baseline itself: those three steps, in that order,
  as one call. No features, no table.
- ``features`` - the described, catalogued measurements and the pipeline
  they are defined against.

The sharing is deliberate and only at this level: ``features`` decodes and
resizes through :mod:`decode` and :mod:`resize`; :mod:`pixels` composes the three
plain steps; neither reaches into the other. The kernel builds no tables, so the
caller names the columns with :func:`flatten.flatten`'s names.
"""
