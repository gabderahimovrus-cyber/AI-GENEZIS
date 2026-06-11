# AI Genesis

AI Genesis is a local-first autonomous AI system scaffold. It is designed to run without cloud APIs and now includes scalable infrastructure for configuration, model lifecycle management, independent queues, local memories, data quality control, metrics storage, benchmarking, and a multi-tab GUI.

## Architecture

```text
AI Genesis
├── configs/                  # YAML configuration for model, training, memory, internet, GUI
├── ai_genesis/model          # Transformer, tokenizer, Model Manager, export .pth/.onnx
├── ai_genesis/training       # Training loop, quality gate, metrics persistence
├── ai_genesis/data           # Dataset Builder, Dataset Registry, Corpus Manager, quality analyzer
├── ai_genesis/internet       # Safe Internet Learning Engine with whitelist, robots.txt, rate limit
├── ai_genesis/memory         # Episodic, Semantic, Learning memory in SQLite
├── ai_genesis/knowledge      # Vector store and SQLite Knowledge Graph triples
├── ai_genesis/benchmark      # Math, logic, programming, history, general knowledge tests
├── ai_genesis/queue          # Independent queues for data, training, eval, chat, system jobs
├── ai_genesis/system         # CPU/RAM/GPU/VRAM monitoring for GUI
├── ai_genesis/teacher        # Teacher, curriculum learning, self-play dataset curation
└── ai_genesis/gui            # Tabs: Chat, Training, Model, Memory, Datasets, Metrics, Logs, Settings
```

## Configuration

All major parameters are loaded from separate YAML files instead of being hardcoded:

- `configs/model.yaml`
- `configs/training.yaml`
- `configs/memory.yaml`
- `configs/internet.yaml`
- `configs/gui.yaml`

The Python config layer keeps safe defaults so tests and local development still work if a config file is missing.

## Model Versioning

Model artifacts are organized as:

```text
models/base
models/candidate
models/production
models/archive
```

`ModelManager` can load production/candidate checkpoints, save new candidate versions, promote a candidate to production, roll back to an archived production version, export `.pth`/`.onnx`, and maintain `models/history.jsonl` plus per-version metadata.

## Data, Metrics, and Quality

- `DatasetRegistry` stores dataset name, creation date, document count, token count, size, source count, language, and quality score.
- `CorpusManager` stores raw source texts, source URL/name, received date, language, token count, corpus versions, and history.
- `DatasetQualityAnalyzer` checks duplicates, language, document length, noise, garbage text, and blocks training below the configured threshold.
- `metrics.db` stores epoch, step, loss, validation loss, perplexity, learning rate, processed tokens, and elapsed training time.

## GUI

Launch the GUI with one of:

```bash
./start_ai_genesis.sh
python run_genesis.py --gui
python -m ai_genesis.main --gui
```

The GUI exposes tabs for Chat, Training, Model, Memory, Datasets, Metrics, Logs, and Settings. It displays model name, version, parameters, size, CPU/RAM/GPU/VRAM status, training status, epoch, loss, perplexity, datasets, recent documents, and a local loss chart.

## CLI

```bash
python -m ai_genesis.main --init-db
python -m ai_genesis.main --gui
python -m ai_genesis.main --collect-data https://en.wikipedia.org/wiki/Transformer_(machine_learning)
python -m ai_genesis.main --train data/datasets/train.jsonl
python -m ai_genesis.main --evaluate
python -m ai_genesis.main --export-model exports/genesis.pth exports/genesis.onnx
```

## Safety Constraints

AI Genesis is local-only by design. It does not use cloud APIs, does not modify system files, does not execute arbitrary operating-system commands from learning modules, does not download or run third-party programs, respects `robots.txt`, rate-limits requests, and only accepts URLs from the configured educational/technical whitelist.
