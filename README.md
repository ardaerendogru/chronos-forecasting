# Chronos Distillation and Classification

This repository contains two primary components that leverage Chronos models for time series tasks:

1. **Distillation Training:**  
   A training script that uses knowledge distillation to transfer knowledge from a large teacher model to a smaller student model. The implementation is found in [scripts/training/distillation.py](scripts/training/distillation.py).

2. **Classification Experiments:**  
   A classification pipeline that utilizes a Chronos model to extract features from time series data and then trains a classifier (or CatBoost model) for various UCR/UEA classification datasets. See [scripts/classification/classification.py](scripts/classification/classification.py) for the full implementation.


## Installation

Clone the repository and install the dependencies via pip:
```bash
git clone https://github.com/ardaerendogru/chronos-forecasting.git
cd chronos-forecasting
pip install -r requirements.txt
```

---

## Usage

### Distillation Training

The distillation script trains a student model using both task-specific loss and a distillation loss derived from a frozen teacher model. It uses a custom `DistillationTrainer` that extends the HuggingFace Trainer and supports multiple hyperparameters.

A sample YAML configuration is provided in [scripts/training/configs/distillation.yaml](scripts/training/configs/distillation.yaml). To start training with distillation, run:

```bash
python scripts/training/distillation.py --config scripts/training/configs/distillation.yaml
```

Key configuration parameters include:
- **model_id:** Identifier for the student model (e.g., "google/t5-efficient-tiny").
- **teacher_id / teacher_model_path:** Identifier/path for the teacher model (e.g., "amazon/chronos-t5-small" or "amazon/chronos-t5-mini").
- **distillation_temperature:** Temperature to scale logits for distillation.
- **learning_rate, batch_size, context_length, prediction_length, etc.:** Additional hyperparameters for training.

### Classification Experiments

The classification experiments script loads a Chronos model and uses it to extract features from time series data (from UCR/UEA datasets). A simple neural network classification head is then trained on these features. The script also supports experiments with CatBoost for a non-deep-learning alternative.

Run the classification experiments as follows:

```bash
python scripts/classification/classification.py
```

Within the script, various experiment settings are looped over:
- **Datasets:** e.g., `["ECG5000", "UWaveGestureLibraryX", "FordA"]`
- **Model Sizes:** e.g., `"tiny"`, `"mini"`, and `"small"`
- **Finetuning Options:** either fine-tuning the encoder or freezing it

Results and logs are written to a file called `results.txt`.

---

## Configuration Details

### Distillation Example Configuration (YAML)

Below is an example from [scripts/training/configs/distillation.yaml](scripts/training/configs/distillation.yaml):

```yaml
training_data_paths:
- "./data/tsmixup-data-10percent.arrow"
- "./kernelsynth-data-10percent.arrow"
probability:
- 0.9
- 0.1
context_length: 512
prediction_length: 64
min_past: 60
max_steps: 100_000
save_steps: 25_000
log_steps: 250
per_device_train_batch_size: 32
learning_rate: 0.001
optim: adamw_torch_fused
num_samples: 20
shuffle_buffer_length: 100_000
gradient_accumulation_steps: 1
model_id: google/t5-efficient-tiny
model_type: seq2seq
random_init: true
tie_embeddings: true
output_dir: ./output/
tf32: false
torch_compile: true
tokenizer_class: "MeanScaleUniformBins"
tokenizer_kwargs:
  low_limit: -15.0
  high_limit: 15.0
n_tokens: 4096
lr_scheduler_type: linear
warmup_ratio: 0.0
dataloader_num_workers: 1
max_missing_prop: 0.9
use_eos_token: true

teacher_id: amazon/chronos-t5-small
distillation_temperature: 2.0
task_loss_weight: 0.5
distill_loss_weight: 0.5
encoder_loss_weight: 0.0025
decoder_loss_weight: 0.025


```

### Classification Configuration

The classification script is self-contained and allows you to adjust parameters directly within the code. Key configurable parameters include:
- **Dataset names:** (e.g., `"ECG5000"`, `"UWaveGestureLibraryX"`, `"FordA"`)
- **Chronos model sizes:** (e.g., `"tiny"`, `"mini"`, `"small"`)
- **Finetuning settings:** Whether to fine-tune the Chronos encoder alongside the classification head
- **Hyperparameters:** dropout rates, batch sizes, number of training epochs, etc.

---

## Project Structure

```
.
├── scripts
│   ├── training
│   │   ├── distillation.py
│   │   └── configs
│   │       └── distillation.yaml
│   └── classification
│       └── classification.py
└── README.md
```


