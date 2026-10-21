# SynReasoner

Chemical-language reasoning for retrosynthesis, forward reaction prediction, and paired-reactant design.

The implementation lives in two standalone scripts: **`train.py` contains the complete training workflow**, and **`test.py` contains inference, chemical evaluation, and self-consistency voting**. There is no internal `src` package to navigate. The original prompt templates, training serialization, chemical normalization, fingerprint scoring, and voting algorithms are retained.

## Reproduce an inference run

Install the dependencies, then run the test script from this directory:

```bash
cd SynReasoner
python -m pip install -e .
python test.py inspect --config configs/local.json
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/local.json
```

The editable install is optional if the dependencies are already available. `python test.py` defaults to inference followed by evaluation. Both scripts accept `--help`.

Set `data` and `model` in `configs/local.json` to your Arrow dataset and checkpoint directories before running the commands above. The example preset selects **three rows**, with greedy decoding and a 5,000-token generation budget. To run a full benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --data /path/to/arrow_dataset \
  --model /path/to/checkpoint --limit 0 --max-tokens 5000 --output runs/uspto_full
```

A parent directory containing one checkpoint is resolved automatically. If it contains multiple checkpoints, specify the exact checkpoint directory. Datasets and model weights must be supplied separately.

## Train a model

All data loading, prompt formatting, Unsloth model setup, fine-tuning, logging, checkpointing, and final-model export are implemented in `train.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data /path/to/train_data \
  --model /path/to/base_model --epochs 20 --batch-size 64 --output runs/training
```

The training CLI also accepts `--gradient-accumulation-steps`, `--learning-rate`, `--max-seq-length`, and `--resume-from-checkpoint`. It saves checkpoints at each epoch and exports the final model together with its tokenizer. Exported Qwen2 checkpoints retain default RoPE fields for both Transformers 5 and older inference readers. WandB is optional; local logs are used when it is unavailable. Training uses the current TRL `SFTConfig` interface.

Training data must contain string columns `question`, `cot`, and `answer`, stored as Arrow shards. Each example is serialized as `question<think>cot</think><answer>answer</answer>` plus the tokenizer EOS token. Supply separate training and evaluation datasets. Additional training settings are defined inside `train()` in `train.py`. Importing the script does not launch training.

## Evaluate and aggregate predictions

| Invocation | Behavior |
| --- | --- |
| `python test.py [options]` | Generate predictions and evaluate the first candidate |
| `python test.py inspect [options]` | Inspect dataset schema, task, checkpoint, and tokenizer |
| `python test.py evaluate ...` | Score existing first-choice TXT predictions |
| `python test.py vote ...` | Normalize and aggregate JSONL candidates into Top-K results |
| `python train.py ...` | Run supervised fine-tuning |

```bash
python test.py evaluate --data runs/inference/ground_truth \
  --predictions runs/inference/predictions.txt --output runs/rescored.json
python test.py vote --data /path/to/test_data \
  --predictions model_a.jsonl model_b.jsonl --topk 10 --workers 1 --output runs/vote
```

TXT predictions contain one first-choice result per line, with embedded newlines escaped. JSONL predictions use a `text` field containing a list of full prompt-plus-generation strings. Input file order defines voting priority when candidates tie.

Transformers is the default inference backend. Select `--backend vllm` after installing vLLM; `--beam-width` enables beam search. `--batch-size` controls the Transformers batch size or the vLLM maximum number of concurrent sequences; the default remains one for Transformers. Both backends use explicit prompt token IDs without an extra BOS token. Batched runs save partial JSONL predictions and progress records. Sampling and precision settings can also be supplied in the JSON preset: `temperature`, `top_p`, `top_k`, `adjust_decimal_places`, `c_shift_decimal_place`, and `h_shift_decimal_place`. Qwen2/DeepSeek inference continues `question<think>`; Qwen3 uses the original chat-template branch.

The inherited scorer defaults to removing chirality and treats Morgan fingerprint Tanimoto 1 as an equivalent retrosynthesis answer. This accuracy is not strict string equality. `infer` evaluates the first candidate; use `vote` for Top-K aggregation.

## File map

```text
SynReasoner/
  train.py               # all training logic and its CLI
  test.py                # all inference, scoring, voting, and their CLI
  configs/local.json     # external assets and decoding settings
  environment.yml        # original training/vLLM dependency pins
  pyproject.toml         # optional installation and inference dependencies
  tests/test_chemistry.py
```

Inside `test.py`, implementation sections are labeled `METRICS`, `GENERATION`, `VOTING`, and `COMMAND-LINE WORKFLOWS`. Conflicting helper names receive `generation_` or `voting_` prefixes so the original algorithms retain separate meanings within one Python namespace. Historical single/multi-checkpoint functions are also defined in this file. The default evaluator path points to `test.py` itself.

Optional installation provides `synreasoner` for the test workflow and `synreasoner-train` for training. Direct script execution requires no project installation.

## Environment and checks

For the original training/vLLM stack:

```bash
conda env create -f environment.yml
conda activate synreasoner
pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Unsloth, TRL, bitsandbytes, and vLLM are required only for their respective workflows. The environment file retains the upstream version pins and adds RDKit.

Each inference run saves selected Arrow examples, `predictions.jsonl`, `predictions.txt`, `metrics.json`, and `validation.json` under its output directory. The validation record includes checkpoint paths, row counts, generation lengths, software versions, elapsed time, and scores.

Supply datasets and model weights separately. Runtime outputs and logs are ignored by Git.
