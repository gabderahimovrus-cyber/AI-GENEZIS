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

from ai_genesis.benchmark.suite import BenchmarkSuite
from ai_genesis.config import InternetConfig, TrainingConfig, load_config
from ai_genesis.data.registry import DatasetQualityAnalyzer
from ai_genesis.internet.learning import InternetLearningEngine


def test_yaml_configs_load_architecture_defaults():
    config = load_config()
    assert config.model.model_name == "Genesis-v1"
    assert isinstance(config.training, TrainingConfig)
    assert config.training.min_new_tokens_for_training > 0
    assert config.internet.allowed_domains


def test_quality_analyzer_blocks_duplicate_garbage_data():
    analyzer = DatasetQualityAnalyzer()
    report = analyzer.analyze(["xxxx" * 20, "xxxx" * 20], expected_language="en")
    assert report["quality_score"] < 0.65
    assert report["passed"] is False


def test_internet_learning_uses_domain_whitelist():
    engine = InternetLearningEngine(InternetConfig(allowed_domains=["example.edu"]))
    assert engine.is_whitelisted("https://docs.example.edu/course")
    assert not engine.is_whitelisted("https://malicious.test/course")


def test_benchmark_suite_scores_answers():
    suite = BenchmarkSuite()
    metrics = suite.run(lambda prompt: "42 animal def Washington Mars")
    assert metrics["accuracy"] == 1.0


def test_online_fallback_answers_without_local_tokenizer_or_checkpoint(tmp_path: Path):
    from ai_genesis.chat.engine import ChatEngine
    from ai_genesis.config import ModelConfig
    from ai_genesis.memory.sqlite_memory import MemoryStore
    from ai_genesis.model.online import OnlineModelConfig

    class FakeOnlineClient:
        config = OnlineModelConfig(enabled=True, provider="fake", model="fake")

        def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
            assert "Пользователь: привет" in prompt
            return "онлайн ответ"

    config = ModelConfig(online_enabled=True)
    engine = ChatEngine(memory=MemoryStore(tmp_path / "memory.sqlite3"), online_client=FakeOnlineClient(), model_config=config)
    assert engine.answer("привет") == "онлайн ответ"


def test_online_mode_downgrades_local_artifacts_to_warnings(tmp_path: Path):
    from ai_genesis.config import MemoryConfig, ModelConfig
    from ai_genesis.diagnostics import SystemDiagnostics

    model_config = ModelConfig(
        online_enabled=True,
        tokenizer_path=tmp_path / "models" / "base" / "tokenizer.model",
        production_dir=tmp_path / "models" / "production",
        candidate_dir=tmp_path / "models" / "candidate",
        archive_dir=tmp_path / "models" / "archive",
    )
    diagnostics = SystemDiagnostics(model_config, MemoryConfig())
    statuses = {item.name: item.status for item in diagnostics.run()}
    assert statuses["Tokenizer"] == "WARNING"
    assert statuses["Production Model"] == "WARNING"
    assert statuses["Online Model"] in {"OK", "WARNING"}
