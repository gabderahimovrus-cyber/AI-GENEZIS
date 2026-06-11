"""Independent local task queues for data collection, training, evaluation, chat, and maintenance."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, Callable

from ai_genesis.logging_system import logger


class QueueName(str, Enum):
    DATA = "data"
    TRAINING = "training"
    EVALUATION = "evaluation"
    CHAT = "chat"
    SYSTEM = "system"


@dataclass(slots=True)
class QueuedTask:
    queue: QueueName
    name: str
    status: str = "queued"


class TaskQueueSystem:
    """Runs each process class in a separate executor so long jobs do not block chat or GUI."""

    def __init__(self) -> None:
        self.executors = {name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"genesis-{name.value}") for name in QueueName}
        self.events: Queue[QueuedTask] = Queue()

    def submit(self, queue: QueueName, name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        task = QueuedTask(queue=queue, name=name)
        self.events.put(task)
        logger.log(f"Задача поставлена в очередь {queue.value}: {name}")

        def runner() -> Any:
            self.events.put(QueuedTask(queue=queue, name=name, status="running"))
            logger.log(f"Задача запущена: {name}")
            result = function(*args, **kwargs)
            self.events.put(QueuedTask(queue=queue, name=name, status="completed"))
            logger.log(f"Задача завершена: {name}")
            return result

        return self.executors[queue].submit(runner)

    def snapshot(self) -> list[QueuedTask]:
        items: list[QueuedTask] = []
        while not self.events.empty():
            items.append(self.events.get())
        return items

    def shutdown(self) -> None:
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
