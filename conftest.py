"""Drop the widget tests when the interpreter cannot import Orange and Qt.

Kernel tests run in a plain Python; widget tests need the Orange interpreter with
a working Qt binding (offscreen is enough). A module spec is not enough: partial
installs exist, so the imports are attempted.
"""


def _orange_usable() -> bool:
    try:
        import AnyQt.QtWidgets  # noqa: F401
        import Orange  # noqa: F401

        return True
    except Exception:
        return False


collect_ignore = [] if _orange_usable() else ["tests/widgets"]
