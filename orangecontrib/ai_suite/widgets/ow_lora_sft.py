"""LoRA SFT: supervised fine-tuning on prompt/response pairs.

A thin shell over :mod:`orangecontrib.ai_suite.kernel.lora`; the shared training UI lives in
:mod:`._lora_base`. This widget supplies the supervised data contract: prompt
and completion columns, with the prompt masked out of the loss.
"""

from __future__ import annotations

from orangecontrib.ai_suite.adapters import table_pairs

from ._lora_base import LoRATrainerBase


class OWLoRASFT(LoRATrainerBase):
    """Supervised fine-tuning of a pretrained language model.

    Input is a Table with prompt and completion text columns (Q&A pairs are a
    special case). Output is a ``LoRAModel``: the base model plus the trained
    adapter. The completion tokens carry the loss and the prompt is masked out,
    so this is supervised fine-tuning rather than continued pre-training.
    Because the contract is just prompt/response pairs, the same widget covers
    Q&A, instruction following and tool use.
    """

    name = "LoRA SFT"
    description = (
        "Supervised fine-tuning on prompt/response pairs: Q&A, tool use, "
        "instruction following (4-bit QLoRA, Apple Silicon)."
    )
    icon = "icons/lora_sft.svg"
    priority = 16
    keywords = [  # noqa: RUF012
        "lora",
        "sft",
        "supervised fine-tuning",
        "q&a",
        "tool use",
        "mlx",
        "qlora",
    ]

    data_box_title = "Data (prompt / completion)"
    data_help = (
        "Rows need prompt and completion text columns. The prompt is masked "
        "out of the loss; only the completion is trained. Use it for Q&A, "
        "instruction following or tool use."
    )

    def _make_lora_data(self):
        from orangecontrib.ai_suite.kernel.lora import LoraData

        return LoraData.from_pairs(table_pairs(self.data))


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLoRASFT).run()
