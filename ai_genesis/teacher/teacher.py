"""Rule-based Teacher AI that creates local training prompts and evaluations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TeachingTask:
    topic: str
    question: str
    expected_traits: list[str]


class TeacherSystem:
    """Assists dataset creation without replacing model training."""

    def suggest_topics(self) -> list[str]:
        return [
            "Python basics",
            "machine learning terminology",
            "Linux safety rules",
            "data cleaning",
            "Transformer architecture",
        ]

    def create_questions(self, topic: str, count: int = 5) -> list[TeachingTask]:
        return [
            TeachingTask(
                topic=topic,
                question=f"Explain {topic} concept #{index + 1} in simple language.",
                expected_traits=["clear", "accurate", "concise"],
            )
            for index in range(count)
        ]

    def score_answer(self, question: str, answer: str) -> dict[str, float | str]:
        length_score = min(len(answer.split()) / 80.0, 1.0)
        relevance_score = 1.0 if any(word.lower() in answer.lower() for word in question.split()[:4]) else 0.4
        score = round((length_score + relevance_score) / 2.0, 3)
        return {"score": score, "feedback": "Use as training data if score is high; revise if low."}
