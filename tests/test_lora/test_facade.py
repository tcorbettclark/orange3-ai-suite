"""The package facade: the names a caller needs, and nothing that is not.

``EXPECTED`` is the whole surface, compared against ``lora.__all__``.
"""

import orangecontrib.ai_suite.kernel.lora as lora

EXPECTED = frozenset(
    {
        "LoRAAdapter",
        "LoRAModel",
        "LoRAData",
        "LoRAOptions",
        "Params",
        "TrainReport",
        "TrainingRun",
        "TrainingStopped",
        "ValReport",
        "available",
        "split_for_training",
        "train_lora",
    }
)


def test_package_exports_only_the_use_case():
    assert set(lora.__all__) == EXPECTED
    for name in lora.__all__:
        assert hasattr(lora, name), name


def test_internals_live_in_their_own_module():
    # Supported, but not part of the façade: the MLX boundary (its wire format,
    # its run) and the domain's own helpers belong to whoever owns the run.
    from orangecontrib.ai_suite.kernel.lora import mlx
    from orangecontrib.ai_suite.kernel.lora.mlx import protocol
    from orangecontrib.ai_suite.kernel.lora.model import _echo
    from orangecontrib.ai_suite.kernel.lora.training import _run_spec

    assert callable(_echo)
    assert callable(protocol.encode) and callable(protocol.decode)
    assert callable(_run_spec)
    assert isinstance(mlx.Run, type)
