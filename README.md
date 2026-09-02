# F5-LoRA

Parameter-efficient fine-tuning for [F5-TTS](https://github.com/SWivid/F5-TTS),
with support for LoRA adapters and full-parameter training.

F5-LoRA is a PyTorch Lightning implementation for adapting flow-matching text-to-speech
models to new speakers and styles. It supports Hugging Face and local audio datasets,
distributed training, SafeTensors checkpoints, and loading or swapping multiple LoRA adapters.

> [!NOTE]
> This is a research-oriented implementation. APIs and checkpoint formats may change as the
> project evolves.

## Features

- **LoRA fine-tuning** for lightweight speaker and style adaptation
- **Full-parameter fine-tuning** when maximum adaptation capacity is required
- **Hugging Face datasets** in streaming or map-style mode
- **Local datasets** described by JSONL or CSV manifests
- **PyTorch Lightning training** with multi-device support
- **SafeTensors checkpoints** for models and adapters
- **Adapter management** for loading, saving, and switching between LoRA adapters

## Audio examples

| Experiment | Reference | Generated output |
| --- | --- | --- |
| Voice effects | [Reference audio](https://drive.google.com/file/d/1kIOpk547k_pXHdTM9U9o51HoWpWQ8D9U/view?usp=sharing) | [Generated audio](https://drive.google.com/file/d/1bota299qYpjzEx9y7eEb1tCeyRC1_Tf1/view?usp=sharing) |
| Voice cloning | [Reference audio](https://drive.google.com/file/d/1w1stYj_g_NbxB30gpN7RNPNxlJgo2gGd/view?usp=sharing) | [Generated audio](https://drive.google.com/file/d/1G6hRLSQRF6LMrIc96_0fPj0rMrKngTR-/view?usp=sharing) |

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/odunola499/f5-lora.git
cd f5-lora
python -m pip install -e .
```

The base F5-TTS checkpoint, vocabulary, and Vocos vocoder are downloaded from Hugging Face
the first time they are needed.

## Quick start

### Train a LoRA adapter

The example below fine-tunes F5-TTS on a Hugging Face audio dataset. Adjust the column names to
match your dataset.

```python
from f5_lora.config import Config, HFData
from f5_lora.train import TrainModule, get_loader, train_model

config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3_000
config.train.warmup_steps = 300
config.train.batch_size = 1
config.train.grad_accumulation_steps = 4
config.train.save_interval = 50

data = HFData(
    repo_id="ylacombe/expresso",
    name=None,
    split="train",
    text_column="text",
    audio_column="audio",
    stream=False,
)

train_loader = get_loader(config.train.batch_size, config, data)
module = TrainModule(
    config,
    train_loader,
    lora=True,
    rank=16,
    alpha=32,
)

train_model(config=config, train_module=module)
```

Adapter weights are written to `checkpoints/` as SafeTensors files. Lightning trainer state is
written separately to `train_checkpoints/` so interrupted runs can be resumed.

### Train on local audio

Local datasets can be described with a JSONL manifest:

```json
{"audio": "speaker/audio_001.wav", "text": "The first training sentence."}
{"audio": "speaker/audio_002.wav", "text": "Another sentence from the same speaker."}
```

Create the corresponding loader with:

```python
from f5_lora.config import Config, LocalData
from f5_lora.train.dataset import get_local_loader

config = Config()
data = LocalData(
    manifest_file="data/manifest.jsonl",
    audio_dir="data/audio",
    audio_column="audio",
    text_column="text",
)

train_loader = get_local_loader(config.train.batch_size, config, data)
```

CSV manifests are also supported. Audio is resampled to 24 kHz and limited to 30 seconds per
training example.

### Generate speech with an adapter

```python
import soundfile as sf
import torch

from f5_lora.config import Config
from f5_lora.infer.inference import Inference
from f5_lora.modules.commons import load_model
from f5_lora.modules.lora import LoraManager

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = load_model(device=device, config=config, dtype=dtype).eval()
manager = LoraManager(model)
manager.load("checkpoints/adapter_3000.safetensors", name="speaker")

inference = Inference(config, model=manager.model.to(device=device, dtype=dtype))
audio, sample_rate, _ = inference(
    ref_audio="reference.wav",
    ref_text="The transcript of the reference recording.",
    gen_text="The sentence to synthesize in the adapted voice.",
)

sf.write("output.wav", audio, sample_rate)
```

Use a clean reference clip no longer than 12 seconds, together with an accurate transcript.

### Full-parameter fine-tuning

Use the same data pipeline and omit the LoRA arguments:

```python
module = TrainModule(config, train_loader)
train_model(config=config, train_module=module)
```

Full-parameter training requires substantially more memory and stores complete model
checkpoints instead of small adapter files.

## Adapter management

`LoraManager` provides a small API for working with one or more adapters:

```python
from f5_lora.modules.lora import LoraManager

manager = LoraManager(model)
manager.prepare(rank=16, alpha=32)
manager.save("speaker.safetensors")

manager.load("speaker.safetensors", name="speaker")
manager.load("style.safetensors", name="style", activate=False)
manager.swap("style")
manager.delete("speaker")
manager.reset()
```

LoRA is currently applied to every linear layer in the model. Rank and alpha are configurable;
the best values depend on the dataset and adaptation task.

## Examples

More complete scripts are available in [`tutorials/`](tutorials), including LoRA training,
full-parameter training, adapter inference, and dataset preparation.

Pretrained experimental adapters are available on
[Hugging Face](https://huggingface.co/odunola/pretrained_adapters/tree/main).

## Responsible use

Voice adaptation and cloning can be misused. Only train on audio you have permission to use,
obtain consent before cloning a person's voice, and clearly disclose synthetic audio where
appropriate. The upstream F5-TTS model and checkpoints remain subject to their respective terms.

## Acknowledgments

- [F5-TTS](https://github.com/SWivid/F5-TTS) for the original model and implementation
- [LoRA](https://arxiv.org/abs/2106.09685) for low-rank adaptation
- [Hugging Face](https://huggingface.co/) for model and dataset tooling
