# pkgutil-style namespace package, so several Orange add-ons can share
# the 'orangecontrib' namespace inside one Python environment.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)