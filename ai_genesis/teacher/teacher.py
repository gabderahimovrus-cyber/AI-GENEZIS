"""Rule-based Teacher AI for curriculum learning and self-play dataset curation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class TeachingTask:
    topic: str
    question: str
    expected_traits: list[str]
    level: str = "facts"


class TeacherSystem:
    """Creates tasks and evaluates answers without editing code or model weights directly."""

    curriculum_levels = ["words", "sentences", "facts", "logic", "dialogues"]

    def suggest_topics(self) -> list[str]:
        return [
            "Python basics",
            "machine learning terminology",
            "Linux safety rules",
            "data cleaning",
            "Transformer architecture",
        ]

    def create_questions(self, topic: str, count: int = 5, level: str = "facts") -> list[TeachingTask]:
        templates = {
            "words": "Define the key term from {topic} #{index}.",
            "sentences": "Write one clear sentence about {topic} #{index}.",
            "facts": "Explain {topic} concept #{index} in simple language.",
            "logic": "Reason step-by-step about a practical {topic} problem #{index}.",
            "dialogues": "Answer a user dialogue question about {topic} #{index}.",
        }
        template = templates.get(level, templates["facts"])
        return [TeachingTask(topic=topic, question=template.format(topic=topic, index=index + 1), expected_traits=["clear", "accurate", "concise"], level=level) for index in range(count)]

    def curriculum(self, topic: str, per_level: int = 3) -> list[TeachingTask]:
        tasks: list[TeachingTask] = []
        for level in self.curriculum_levels:
            tasks.extend(self.create_questions(topic, per_level, level=level))
        return tasks

    def score_answer(self, question: str, answer: str) -> dict[str, float | str]:
        length_score = min(len(answer.split()) / 80.0, 1.0)
        relevance_score = 1.0 if any(word.lower().strip("?.!,") in answer.lower() for word in question.split()[:4]) else 0.4
        clarity_score = 0.8 if len(answer.strip()) > 20 and "\x00" not in answer else 0.2
        score = round((length_score + relevance_score + clarity_score) / 3.0, 3)
        return {"score": score, "feedback": "Use as training data if score is high; revise if low."}

    def self_play(self, genesis_answer: Callable[[str], str], topic: str, output_path: Path, count: int = 10, min_score: float = 0.65) -> list[dict[str, object]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        accepted: list[dict[str, object]] = []
        for task in self.curriculum(topic, per_level=max(1, count // len(self.curriculum_levels)))[:count]:
            answer = genesis_answer(task.question)
            score = self.score_answer(task.question, answer)
            record = {"task": asdict(task), "answer": answer, "score": score["score"]}
            if float(score["score"]) >= min_score:
                accepted.append(record)
        with output_path.open("a", encoding="utf-8") as file:
            for record in accepted:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return accepted
