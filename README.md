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


## First-run MVP initialization

On the first GUI launch AI Genesis now checks the complete local runtime: `tokenizer.model`, `tokenizer.vocab`, candidate and production checkpoints, datasets, SQLite databases, and required project directories. If anything required is missing, the GUI opens a first-run wizard with one action: **«Инициализировать Genesis»**.

The same workflow is available from CLI:

```bash
python -m ai_genesis.main --init-genesis
```

Initialization creates the directory layout, initializes SQLite stores, writes a demo educational corpus, trains the SentencePiece tokenizer, creates the first JSONL dataset, runs a short smoke-training cycle, saves checkpoints, registers candidate and production versions through `ModelManager`, records metrics, and stores a machine-readable `genesis_initialized.json` summary.

## Diagnostics and hardware monitoring

The GUI includes a **«Диагностика системы»** tab with OK/WARNING/ERROR checks for Tokenizer, Dataset, Production Model, Candidate Model, CUDA, PyTorch, SentencePiece, FAISS, SQLite, Vector Store, Memory System, Teacher System, and Internet Learning Engine.

Hardware monitoring reports CPU load, RAM usage, CUDA device name, VRAM total/used, GPU utilization, and GPU temperature when PyTorch CUDA and/or `nvidia-smi` are available. If CUDA is missing, diagnostics shows a clear PyTorch CUDA installation hint.

## Teacher and local assistant models

The Teacher module remains separate from Genesis weights and source code. It can generate curriculum tasks, score answers, curate question-answer records, and identify topics for further learning. The GUI Teacher tab also supports optional helper models as Teacher or Knowledge Assistant through:

- Hugging Face Transformers (`transformers` backend)
- GGUF via `llama-cpp-python` (`gguf` backend)
- Ollama local server (`ollama` backend)

These assistants can help create training data and evaluations while Genesis remains the project-owned model trained through the local pipeline.

## GUI

Launch the GUI with one of:

```bash
./start_ai_genesis.sh
python run_genesis.py --gui
python -m ai_genesis.main --gui
```

The GUI exposes tabs for Chat, Training, Teacher, Models, System Diagnostics, Memory, Datasets, Metrics, Logs, and Settings. It displays model name, version, parameters, size, CPU/RAM/GPU/VRAM status, training status, epoch, loss, perplexity, datasets, recent documents, and a local loss chart.

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
