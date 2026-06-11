"""Tkinter GUI for AI Genesis."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from ai_genesis.chat.engine import ChatEngine
from ai_genesis.config import ModelConfig
from ai_genesis.logging_system import logger


class GenesisGUI(tk.Tk):
    """Main window with status, chat, logs, and training controls."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AI Genesis")
        self.geometry("1200x720")
        self.chat_engine = ChatEngine()
        self.model_config = ModelConfig()
        self._build_layout()
        self._refresh_logs()

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = tk.Frame(self, width=220, padx=10, pady=10)
        left.grid(row=0, column=0, sticky="ns")
        tk.Label(left, text="Статус системы", font=("Arial", 14, "bold")).pack(anchor="w")
        self.status_label = tk.Label(
            left,
            justify="left",
            text="Модель:\nGenesis-v1\n\nПараметры:\n25M\n\nУстройство:\nGTX 1650\n\nСтатус:\nОжидание",
        )
        self.status_label.pack(anchor="w", pady=12)
        center = tk.Frame(self, padx=10, pady=10)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.chat_view = scrolledtext.ScrolledText(center, wrap=tk.WORD)
        self.chat_view.grid(row=0, column=0, sticky="nsew")
        entry_frame = tk.Frame(center)
        entry_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        entry_frame.columnconfigure(0, weight=1)
        self.message_entry = tk.Entry(entry_frame)
        self.message_entry.grid(row=0, column=0, sticky="ew")
        self.message_entry.bind("<Return>", lambda _: self._send_message())
        tk.Button(entry_frame, text="Отправить", command=self._send_message).grid(row=0, column=1, padx=5)
        tk.Button(entry_frame, text="Сохранить диалог", command=self._save_dialogue).grid(row=0, column=2)
        right = tk.Frame(self, width=300, padx=10, pady=10)
        right.grid(row=0, column=2, sticky="ns")
        tk.Label(right, text="Лог работы", font=("Arial", 14, "bold")).pack(anchor="w")
        self.log_view = scrolledtext.ScrolledText(right, width=38, height=30, wrap=tk.WORD)
        self.log_view.pack(fill="both", expand=True)
        controls = tk.Frame(self, padx=10, pady=10)
        controls.grid(row=1, column=0, columnspan=3, sticky="ew")
        for label, command in [
            ("Начать обучение", self._start_training),
            ("Пауза", lambda: logger.log("Пауза запрошена")),
            ("Остановить", lambda: logger.log("Остановка запрошена")),
            ("Сохранить модель", lambda: logger.log("Сохранение модели запрошено")),
            ("Загрузить модель", self._load_model),
            ("Очистить лог", self._clear_log),
        ]:
            tk.Button(controls, text=label, command=command).pack(side="left", padx=4)

    def _send_message(self) -> None:
        message = self.message_entry.get().strip()
        if not message:
            return
        self.message_entry.delete(0, tk.END)
        self.chat_view.insert(tk.END, f"Пользователь:\n{message}\n\n")
        threading.Thread(target=self._answer_async, args=(message,), daemon=True).start()

    def _answer_async(self, message: str) -> None:
        answer = self.chat_engine.answer(message)
        self.after(0, self._append_answer, answer)

    def _append_answer(self, answer: str) -> None:
        self.chat_view.insert(tk.END, f"Genesis:\n{answer}\n\n")
        logger.log("Ответ Genesis добавлен в чат")

    def _save_dialogue(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(self.chat_view.get("1.0", tk.END))
            logger.log(f"Диалог сохранён: {path}")

    def _start_training(self) -> None:
        self.status_label.configure(
            text="Модель:\nGenesis-v1\n\nПараметры:\n25M\n\nУстройство:\nGTX 1650\n\nСтатус:\nОбучение"
        )
        logger.log("Начато обучение")

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoints", "*.pt")])
        if path:
            self.chat_engine.load_model(path)
            logger.log(f"Модель загружена: {path}")

    def _clear_log(self) -> None:
        logger.clear()
        self.log_view.delete("1.0", tk.END)

    def _refresh_logs(self) -> None:
        self.log_view.delete("1.0", tk.END)
        self.log_view.insert(tk.END, "\n".join(logger.snapshot()))
        self.after(1000, self._refresh_logs)


def run_gui() -> None:
    app = GenesisGUI()
    app.mainloop()
