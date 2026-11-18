"""Image data: what the caller handed us, and what can be measured from it.

Four parts, each with a reason to change of its own:

- :mod:`decode` - what the caller handed us: alpha, channel order, value
  range. Parameterless, and it refuses a range it cannot interpret.
- :mod:`resize` - resampling to a fixed square, one policy per axis: area
  averaging when an axis shrinks, bilinear when it grows.
- :mod:`flatten` - a 2-D grid as one row, with a coordinate name per cell:
  the columns of a pixel-values table.
- ``features`` - the described, catalogued measurements and the pipeline
  they are defined against.

The sharing is deliberate and only at this level: ``features`` decodes and
resizes through :mod:`decode` and :mod:`resize`; the widget that composes the
plain pixel-values pipeline does the calling, because the kernel builds no
tables.
"""
