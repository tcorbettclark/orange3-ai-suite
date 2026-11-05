"""LoRA Pre-trainer: continue pre-training a base LLM on raw text.

A thin shell over :mod:`orangecontrib.ai_suite.kernel.lora`; the shared training UI lives in
:mod:`._lora_base`. This widget supplies the continued-pre-training data
contract: one document per row, self-supervised next-token loss.
"""

from __future__ import annotations

from orangecontrib.ai_suite.adapters import table_texts

from ._lora_base import LoRATrainerBase


class OWLoRAPretrainer(LoRATrainerBase):
    """Continue pre-training a pretrained language model on a raw corpus.

    Input is a Table of text documents, one per row. Output is a ``LoRAModel``:
    the base model plus the trained adapter. The corpus is unlabelled and the
    loss is the ordinary next-token objective, so this is still pre-training;
    the adapter only restricts which weights change. Use LoRA SFT when the data
    contains prompt/response pairs.
    """

    name = "LoRA Pre-trainer"
    description = (
        "Continue pre-training a base LLM on a raw text corpus (4-bit QLoRA, Apple Silicon)."
    )
    icon = "icons/lora_pretrainer.svg"
    priority = 15
    keywords = [  # noqa: RUF012
        "lora",
        "continued pre-training",
        "domain adaptation",
        "mlx",
        "qlora",
    ]

    data_box_title = "Data (raw corpus)"
    data_help = (
        "Rows are documents. Training is self-supervised: the model learns "
        "the next token, exactly as in pre-training, so no labels are needed."
    )

    def _make_lora_data(self):
        from orangecontrib.ai_suite.kernel.lora import LoraData

        return LoraData.from_texts(table_texts(self.data))


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLoRAPretrainer).run()
