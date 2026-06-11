"""Tkinter GUI for AI Genesis with tabs for chat, training, model, memory, datasets, metrics, logs, and settings."""

from __future__ import annotations

import json
from dataclasses import asdict
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ai_genesis.chat.engine import ChatEngine
from ai_genesis.config import GUIConfig, ModelConfig, load_config
from ai_genesis.data.registry import DatasetRegistry
from ai_genesis.logging_system import logger
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.metrics.store import MetricsStore
from ai_genesis.model.manager import ModelManager
from ai_genesis.queue.task_queue import QueueName, TaskQueueSystem
from ai_genesis.system.monitor import SystemMonitor


class GenesisGUI(tk.Tk):
    """Main Genesis desktop interface."""

    def __init__(self) -> None:
        super().__init__()
        self.config_bundle = load_config()
        self.gui_config: GUIConfig = self.config_bundle.gui
        self.model_config: ModelConfig = self.config_bundle.model
        self.title(self.gui_config.title)
        self.geometry(f"{self.gui_config.width}x{self.gui_config.height}")
        self.chat_engine = ChatEngine()
        self.monitor = SystemMonitor()
        self.task_queue = TaskQueueSystem()
        self.model_manager = ModelManager(self.model_config)
        self.memory = MemoryStore()
        self.datasets = DatasetRegistry()
        self.metrics = MetricsStore()
        self._build_layout()
        self._refresh_all()

    def _build_layout(self) -> None:
        self.status = tk.StringVar(value="Загрузка статуса...")
        tk.Label(self, textvariable=self.status, justify="left", anchor="w", padx=8).pack(fill="x")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self._build_chat_tab()
        self._build_training_tab()
        self._build_model_tab()
        self._build_memory_tab()
        self._build_datasets_tab()
        self._build_metrics_tab()
        self._build_logs_tab()
        self._build_settings_tab()

    def _tab(self, name: str) -> ttk.Frame:
        frame = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(frame, text=name)
        return frame

    def _build_chat_tab(self) -> None:
        tab = self._tab("Чат")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.chat_view = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.chat_view.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.message_entry = ttk.Entry(tab)
        self.message_entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.message_entry.bind("<Return>", lambda _: self._send_message())
        ttk.Button(tab, text="Отправить", command=self._send_message).grid(row=1, column=1, padx=5, pady=(8, 0))
        ttk.Button(tab, text="Сохранить диалог", command=self._save_dialogue).grid(row=1, column=2, pady=(8, 0))

    def _build_training_tab(self) -> None:
        tab = self._tab("Обучение")
        self.training_text = scrolledtext.ScrolledText(tab, height=18, wrap=tk.WORD)
        self.training_text.pack(fill="both", expand=True)
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="Запустить обучение", command=self._start_training).pack(side="left", padx=4)
        ttk.Button(controls, text="Пауза", command=lambda: logger.log("Пауза обучения запрошена")).pack(side="left", padx=4)
        ttk.Button(controls, text="Остановить", command=lambda: logger.log("Остановка обучения запрошена")).pack(side="left", padx=4)

    def _build_model_tab(self) -> None:
        tab = self._tab("Модель")
        self.model_text = scrolledtext.ScrolledText(tab, height=22, wrap=tk.WORD)
        self.model_text.pack(fill="both", expand=True)
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="Загрузить модель", command=self._load_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Продвинуть candidate", command=self._promote_candidate).pack(side="left", padx=4)
        ttk.Button(controls, text="Откат production", command=self._rollback_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Экспорт .pth/.onnx", command=lambda: logger.log("Экспорт запускается через CLI --export-model")).pack(side="left", padx=4)

    def _build_memory_tab(self) -> None:
        tab = self._tab("Память")
        self.memory_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.memory_text.pack(fill="both", expand=True)

    def _build_datasets_tab(self) -> None:
        tab = self._tab("Датасеты")
        self.datasets_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.datasets_text.pack(fill="both", expand=True)
        ttk.Button(tab, text="Обновить список датасетов", command=self._refresh_datasets).pack(anchor="w", pady=8)

    def _build_metrics_tab(self) -> None:
        tab = self._tab("Метрики")
        self.metrics_canvas = tk.Canvas(tab, height=260, background="white")
        self.metrics_canvas.pack(fill="x")
        self.metrics_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.metrics_text.pack(fill="both", expand=True)

    def _build_logs_tab(self) -> None:
        tab = self._tab("Логи")
        self.log_view = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.log_view.pack(fill="both", expand=True)
        ttk.Button(tab, text="Очистить лог", command=self._clear_log).pack(anchor="w", pady=8)

    def _build_settings_tab(self) -> None:
        tab = self._tab("Настройки")
        self.settings_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.settings_text.pack(fill="both", expand=True)
        self.settings_text.insert(tk.END, json.dumps({
            "model": asdict(self.model_config),
            "training": asdict(self.config_bundle.training),
            "memory": asdict(self.config_bundle.memory),
            "internet": asdict(self.config_bundle.internet),
            "gui": asdict(self.gui_config),
        }, ensure_ascii=False, indent=2, default=str))

    def _send_message(self) -> None:
        message = self.message_entry.get().strip()
        if not message:
            return
        self.message_entry.delete(0, tk.END)
        self.chat_view.insert(tk.END, f"Пользователь:\n{message}\n\n")
        self.task_queue.submit(QueueName.CHAT, "chat-answer", self._answer_and_append, message)

    def _answer_and_append(self, message: str) -> None:
        answer = self.chat_engine.answer(message)
        self.after(0, self._append_answer, answer)

    def _append_answer(self, answer: str) -> None:
        self.chat_view.insert(tk.END, f"Genesis:\n{answer}\n\n")
        logger.log("Ответ Genesis добавлен в чат")

    def _save_dialogue(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            Path(path).write_text(self.chat_view.get("1.0", tk.END), encoding="utf-8")
            logger.log(f"Диалог сохранён: {path}")

    def _start_training(self) -> None:
        self.monitor.update_training("queued")
        self.task_queue.submit(QueueName.TRAINING, "training-cycle", lambda: logger.log("Обучение запускается через CLI --train с датасетом"))

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoints", "*.pt")])
        if path:
            self.chat_engine.load_model(Path(path))
            logger.log(f"Модель загружена: {path}")

    def _promote_candidate(self) -> None:
        messagebox.showinfo("Model Manager", "Candidate продвинута" if self.model_manager.promote_candidate() else "Candidate не найдена")

    def _rollback_model(self) -> None:
        messagebox.showinfo("Model Manager", "Откат выполнен" if self.model_manager.rollback() else "Архив пуст")

    def _clear_log(self) -> None:
        logger.clear()
        self.log_view.delete("1.0", tk.END)

    def _refresh_all(self) -> None:
        self._refresh_status()
        self._refresh_logs()
        self._refresh_memory()
        self._refresh_datasets()
        self._refresh_metrics()
        self.after(self.gui_config.refresh_ms, self._refresh_all)

    def _refresh_status(self) -> None:
        snapshot = self.monitor.snapshot()
        model_status = self.model_manager.current_status()
        self.status.set(
            f"Модель: {self.model_config.model_name} | Версия: {model_status.get('version', 'нет')} | "
            f"Параметры: {model_status.get('parameters', 'неизвестно')} | Размер: {model_status.get('size_bytes', 0)} байт | "
            f"CPU: {snapshot.cpu_percent:.1f}% | RAM: {snapshot.ram_percent:.1f}% ({snapshot.ram_used_gb:.2f} GB) | "
            f"GPU: {snapshot.gpu_name} | VRAM: {snapshot.vram_used_mb}/{snapshot.vram_total_mb} MB | Temp: {snapshot.gpu_temperature_c} | "
            f"Обучение: {snapshot.training_status}, эпоха {snapshot.epoch}, loss={snapshot.loss}, ppl={snapshot.perplexity}"
        )
        self.training_text.delete("1.0", tk.END)
        self.training_text.insert(tk.END, json.dumps(asdict(snapshot), ensure_ascii=False, indent=2))
        self.model_text.delete("1.0", tk.END)
        self.model_text.insert(tk.END, json.dumps({"current": model_status, "history": self.model_manager.history(20)}, ensure_ascii=False, indent=2, default=str))

    def _refresh_logs(self) -> None:
        self.log_view.delete("1.0", tk.END)
        self.log_view.insert(tk.END, "\n".join(logger.snapshot()))

    def _refresh_memory(self) -> None:
        self.memory_text.delete("1.0", tk.END)
        self.memory_text.insert(tk.END, "Episodic Memory:\n")
        for role, content in self.memory.recent_dialogue(20):
            self.memory_text.insert(tk.END, f"{role}: {content}\n")
        self.memory_text.insert(tk.END, "\nSemantic Memory:\n" + "\n".join(self.memory.recent_facts(20)))
        self.memory_text.insert(tk.END, "\n\nLearning Memory:\n" + "\n".join(f"{e}: {m}" for e, m in self.memory.recent_learning_events(20)))

    def _refresh_datasets(self) -> None:
        self.datasets_text.delete("1.0", tk.END)
        for item in self.datasets.list_datasets(50):
            self.datasets_text.insert(tk.END, json.dumps(asdict(item), ensure_ascii=False, indent=2) + "\n")
            docs = self.datasets.recent_documents(Path(item.path), 3)
            if docs:
                self.datasets_text.insert(tk.END, "Последние документы:\n" + "\n---\n".join(docs) + "\n\n")

    def _refresh_metrics(self) -> None:
        rows = self.metrics.recent(self.gui_config.metrics_points)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(json.dumps(asdict(row), ensure_ascii=False, default=str) for row in rows))
        self._draw_loss_chart([row.loss for row in rows])

    def _draw_loss_chart(self, losses: list[float]) -> None:
        self.metrics_canvas.delete("all")
        if len(losses) < 2:
            self.metrics_canvas.create_text(20, 20, anchor="w", text="Недостаточно точек для графика loss")
            return
        width = max(self.metrics_canvas.winfo_width(), 600)
        height = 240
        mn, mx = min(losses), max(losses)
        span = max(mx - mn, 1e-6)
        points = []
        for i, loss in enumerate(losses):
            x = 20 + i * (width - 40) / (len(losses) - 1)
            y = height - 20 - (loss - mn) * (height - 40) / span
            points.extend([x, y])
        self.metrics_canvas.create_line(points, fill="#1565c0", width=2)
        self.metrics_canvas.create_text(20, 10, anchor="w", text=f"Loss: min={mn:.4f}, max={mx:.4f}")


def run_gui() -> None:
    app = GenesisGUI()
    app.mainloop()
