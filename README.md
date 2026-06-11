# AI Genesis

AI Genesis is a local-first MVP for an autonomous learning AI system. It is designed to run without cloud APIs and includes a custom tokenizer, a compact decoder-only Transformer, local memory, data collection, training, evaluation, a knowledge vector store, and a graphical chat interface.

## MVP Components

```text
AI Genesis
├── GUI
├── Chat Engine
├── Neural Network
├── Tokenizer
├── Memory
├── Internet Learning
├── Teacher System
├── Dataset Builder
├── Training System
├── Evaluation System
├── Knowledge Base
├── Vector Store
└── Logging
```

## Implemented Capabilities

- **GUI:** Tkinter window with a system status panel, chat history, log panel, and controls for training, pause, stop, save/load model, and log clearing.
- **Chat Engine:** Local chat pipeline that stores user and Genesis messages in SQLite and can use vector-store context for RAG-style prompts.
- **Neural Network:** PyTorch decoder-only Transformer named `Genesis-v1` with the MVP configuration: `vocab_size=32000`, `hidden_size=256`, `layers=6`, `heads=8`, and `context_length=512`.
- **Tokenizer:** SentencePiece BPE wrapper with training, saving, loading, encoding, and decoding.
- **Internet Learning:** Polite educational content downloader that checks `robots.txt`, uses a project user agent, rate-limits requests, and avoids aggressive crawling.
- **Teacher System:** Rule-based assistant that proposes topics, generates question tasks, and scores candidate answers for dataset curation.
- **Dataset Builder:** Cleans text, removes duplicates, normalizes whitespace, and writes JSONL token datasets.
- **Training System:** Supports scratch training, resume from checkpoint, checkpoint autosave every N steps, and local metrics output.
- **Evaluation System:** Computes validation loss, perplexity, and generation speed.
- **Memory:** SQLite-backed episodic, semantic, and learning memory.
- **Vector Store:** FAISS-backed semantic lookup when FAISS is installed, with a deterministic local NumPy fallback.
- **Versioning Layout:** `models/base`, `models/candidate`, `models/production`, and `models/archive` directories are created for promotion/rollback workflows.

## Quick Start

```bash
python -m ai_genesis.main --init-db
python -m ai_genesis.main --gui
```

## Safety Constraints

The project is structured so learning modules collect text and train local model artifacts, but they do not modify system files, execute arbitrary operating-system commands, download and run programs, or rewrite their own source code without user confirmation.
