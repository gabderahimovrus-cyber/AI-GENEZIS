from pathlib import Path

from ai_genesis.config import ModelConfig
from ai_genesis.data.dataset_builder import DatasetBuilder
from ai_genesis.knowledge.vector_store import VectorStore
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.teacher.teacher import TeacherSystem


def test_model_config_matches_mvp_spec():
    config = ModelConfig()
    assert config.vocab_size == 32_000
    assert config.hidden_size == 256
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.context_length == 512
    assert config.model_name == "Genesis-v1"


def test_dataset_builder_cleans_and_deduplicates(tmp_path: Path):
    builder = DatasetBuilder()
    output = builder.write_clean_corpus(
        [
            "  Long educational text about local neural networks and transformers.  ",
            "Long educational text about local neural networks and transformers.",
            "short",
        ],
        tmp_path / "corpus.txt",
    )
    assert output.read_text(encoding="utf-8").splitlines() == [
        "Long educational text about local neural networks and transformers."
    ]


def test_memory_store_separates_dialogue_fact_and_learning(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_dialogue("user", "Привет")
    store.add_fact("Genesis is local", "test")
    store.add_learning_event("epoch_complete", '{"loss": 1.0}')
    assert store.recent_dialogue() == [("user", "Привет")]


def test_vector_store_searches_relevant_text():
    store = VectorStore(dimension=32)
    store.add_texts(["transformer neural network", "gardening and soil"])
    results = store.search("neural transformer", k=1)
    assert results[0][0] == "transformer neural network"


def test_teacher_creates_questions_and_scores_answers():
    teacher = TeacherSystem()
    tasks = teacher.create_questions("Transformer", count=2)
    assert len(tasks) == 2
    score = teacher.score_answer(tasks[0].question, "Transformer models use attention to process tokens.")
    assert 0 <= score["score"] <= 1
