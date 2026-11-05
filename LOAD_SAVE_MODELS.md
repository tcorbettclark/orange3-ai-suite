# Load/Save Models — temporary decision note

## Current decision

- Removed the inline persistence buttons:
  - **LM Pre-trainer**: Save model
  - **LM Generator**: Load model...
  - **LoRA Pre-trainer** / **LoRA SFT**: Save adapter
- Models travel only through the normal Orange **Model** signal.
- Persistence should return as dedicated **Save LM** and **Load LM** widgets, not as controls inside training and generation widgets.

## Why

- The buttons duplicated the normal Orange model input/output flow.
- A model loaded by the Generator could override or outlive the connected input, making the dataflow unclear.
- Standard Orange **Save Model / Load Model** do not fit:
  - they pickle `Orange.base.Model`
  - our Keras `LanguageModel` and MLX `LoRAModel` are different object types
  - our artifacts are directories (`meta.json` / `lora_meta.json` plus weights), not single pickle files.

## Future Load LM scope

Auto-detect and load:

1. Keras LM directory (`meta.json` + `model.keras`)
2. LoRA adapter directory (`lora_meta.json` + adapter weights)
3. Raw MLX model directory (`config.json` + safetensors), including converted BritLLM 4-bit
4. Optionally a Hugging Face repo id, if `mlx_lm` can load it on the machine

Do **not** promise support for:

- GGUF / llama.cpp models
- Ollama models
- PyTorch-only models
- models too large for available memory
- arbitrary unquantised models

State the supported backends in the widget.

## Future Save LM

- Input: `Model` (`object`)
- Save a Keras LM or LoRA adapter artifact directory.
- Raw base LLMs should not be presented as savable artifacts unless a concrete copy/convert path is added.

## State at time of writing

- Buttons removed in commit `d29dbbc`.
- Tutorials currently state that persistence is moving to Load LM/Save LM.
- No load/save replacement exists yet.
