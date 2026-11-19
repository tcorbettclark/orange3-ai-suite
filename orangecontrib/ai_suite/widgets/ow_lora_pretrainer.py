"""LoRA Pre-trainer: a LoRA adapter trained on a corpus of documents.

One document per row, trained on the ordinary next-token objective. The widget
itself is the shared LoRA training UI in :mod:`._lora_base`.
"""

from __future__ import annotations

from orangecontrib.ai_suite.adapters import table_texts

from ._lora_base import LoRATrainerBase


class OWLoRAPretrainer(LoRATrainerBase):
    """Continue pre-training a pretrained language model on a raw corpus.

    Input is a Table of text documents, one per row. Output is a ``LoRAModel``: the
    base model plus the trained adapter. The corpus is unlabelled and the loss is the
    ordinary next-token objective.
    """

    name = "LoRA Pre-trainer"
    description = (
        "Continue pre-training a base LLM on a raw text corpus (LoRA/QLoRA, Apple Silicon)."
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
        from edu_ai_suite.lora import LoRAData

        return LoRAData.from_texts(table_texts(self.data))


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLoRAPretrainer).run()
