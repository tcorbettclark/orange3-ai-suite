"""Image data: how it is decoded, and what can be measured from it.

Three parts, each with a reason to change of its own:

- :mod:`decode` - what the caller handed us: alpha, channel order, value
  range. Parameterless, and it refuses a range it cannot interpret.
- :mod:`pixel_values` - the plain baseline: an image becomes a small grid of
  luminance values. It is not a catalogue feature.
- ``features`` - the described, catalogued measurements and the pipeline they
  are defined against.

The sharing is deliberate and only at this level: ``features`` and ``pixel_values``
both decode through :mod:`decode`, and neither reaches into the other.
"""
