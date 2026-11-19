"""The child's couplings to mlx-lm, checked against the installed library.

No model, no weights, no Metal work: just the two names the child reaches into
mlx-lm for - its argument parser (for current defaults) and the module global it
replaces to turn the trainable-parameter report into data. An mlx-lm upgrade that
moves either one fails here instead of halfway through a student's run.

Skipped when the ``lora`` extra is not installed, like the rest of the LoRA
kernel tests.
"""

from types import SimpleNamespace

import pytest

mlx_lm = pytest.importorskip("mlx_lm", reason="the lora extra (mlx-lm) is not installed")

from orangecontrib.ai_suite.kernel.lora import _child  # noqa: E402

SPEC = {
    "model": "the-base",
    "data": "/runs/1/data",
    "adapter_path": "/runs/1/adapter",
    "mask_prompt": False,
    "iters": 7,
    "learning_rate": 2e-5,
    "num_layers": 3,
    "batch_size": 2,
    "max_seq_length": 64,
    "steps_per_report": 1,
    "steps_per_eval": 2,
    "save_every": 3,
    "seed": 11,
}


def test_mlx_lm_still_hands_out_a_namespace_of_defaults():
    import mlx_lm.lora as lora

    args = lora.build_parser().parse_args([])
    # the fields apply_spec overwrites, and the one it deliberately leaves alone
    for name in ("fine_tune_type", "adapter_path", "save_every", "mask_prompt"):
        assert hasattr(args, name), name
    assert hasattr(args, "trust_remote_code")


def test_the_spec_fills_that_namespace_into_a_lora_run():
    import mlx_lm.lora as lora

    args = lora.build_parser().parse_args([])
    _child.apply_spec(args, SPEC)
    assert args.train is True and args.test is False
    assert args.fine_tune_type == "lora"
    assert args.iters == 7
    assert args.save_every == 3
    assert args.mask_prompt is False


def test_the_printer_the_params_hook_replaces_is_still_a_module_global():
    import mlx_lm.lora as lora

    original = lora.print_trainable_parameters
    try:
        _child.install_params_hook(lora)
        assert lora.print_trainable_parameters is not original
        assert lora.print_trainable_parameters.__module__ == _child.__name__
    finally:
        lora.print_trainable_parameters = original


def test_the_counts_mlx_lm_prints_are_ones_the_child_can_read():
    # The hook's arithmetic needs both of these to exist; it is the same sum
    # mlx-lm's own printer does, so the metric and the log line agree.
    from mlx.utils import tree_flatten
    from mlx_lm.utils import get_total_parameters

    assert callable(tree_flatten)
    assert callable(get_total_parameters)
    assert get_total_parameters(SimpleNamespace(leaf_modules=lambda: [])) == 0
