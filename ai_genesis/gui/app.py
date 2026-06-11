"""Modern Tkinter GUI for the AI Genesis local MVP."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ai_genesis.chat.engine import ChatEngine
from ai_genesis.config import GUIConfig, ModelConfig, ROOT_DIR, load_config
from ai_genesis.data.registry import DatasetRegistry
from ai_genesis.diagnostics import SystemDiagnostics
from ai_genesis.initialization import GenesisInitializer
from ai_genesis.logging_system import logger
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.metrics.store import MetricsStore
from ai_genesis.model.manager import ModelManager
from ai_genesis.model.tokenizer import GenesisTokenizer
from ai_genesis.queue.task_queue import TaskQueueSystem
from ai_genesis.system.monitor import SystemMonitor
from ai_genesis.teacher.teacher import AssistantModelConfig, LocalAssistantModel, TeacherSystem


class GenesisGUI(tk.Tk):
    """Main Genesis desktop interface with first-run wizard and diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.config_bundle = load_config()
        self.gui_config: GUIConfig = self.config_bundle.gui
        self.model_config: ModelConfig = self.config_bundle.model
        self.title(self.gui_config.title)
        self.geometry(f"{self.gui_config.width}x{self.gui_config.height}")
        self.minsize(1100, 720)
        self.chat_engine = ChatEngine()
        self.monitor = SystemMonitor()
        self.task_queue = TaskQueueSystem()
        self.model_manager = ModelManager(self.model_config)
        self.memory = MemoryStore()
        self.datasets = DatasetRegistry()
        self.metrics = MetricsStore()
        self.diagnostics = SystemDiagnostics(self.model_config, self.config_bundle.memory)
        self.teacher = TeacherSystem()
        self.current_task = tk.StringVar(value="Ожидание")
        self._style()
        self._build_layout()
        self._load_runtime_model()
        self._refresh_all()
        self.after(250, self._show_initializer_if_needed)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f7fb")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#f4f7fb")
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), background="#ffffff")
        style.configure("TButton", padding=8)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header()
        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(body)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._build_chat_tab()
        self._build_training_tab()
        self._build_teacher_tab()
        self._build_model_tab()
        self._build_diagnostics_tab()
        self._build_memory_tab()
        self._build_datasets_tab()
        self._build_metrics_tab()
        self._build_logs_tab()
        self._build_settings_tab()

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="Card.TFrame", padding=12)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        for index in range(8):
            header.columnconfigure(index, weight=1)
        self.model_name_var = tk.StringVar(value="Genesis-v1")
        self.version_var = tk.StringVar(value="Версия: загрузка")
        self.params_var = tk.StringVar(value="Параметры: загрузка")
        self.size_var = tk.StringVar(value="Размер: загрузка")
        self.device_var = tk.StringVar(value="Устройство: CPU")
        self.memory_var = tk.StringVar(value="Память: загрузка")
        self.train_status_var = tk.StringVar(value="Обучение: idle")
        ttk.Label(header, textvariable=self.model_name_var, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        for column, var in enumerate([self.version_var, self.params_var, self.size_var, self.device_var, self.memory_var, self.train_status_var, self.current_task], start=1):
            ttk.Label(header, textvariable=var, style="Header.TLabel").grid(row=0, column=column, sticky="ew", padx=8)

    def _tab(self, name: str) -> ttk.Frame:
        frame = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(frame, text=name)
        return frame

    def _bind_text_actions(self, widget: tk.Widget) -> None:
        menu = tk.Menu(widget, tearoff=False)
        menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: widget.event_generate("<<SelectAll>>"))
        widget.bind("<Control-a>", lambda event: (widget.event_generate("<<SelectAll>>"), "break")[1])
        widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def _build_chat_tab(self) -> None:
        tab = self._tab("Чат")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.chat_view = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=("Segoe UI", 10), padx=10, pady=10)
        self.chat_view.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.chat_view.configure(state="disabled")
        self._bind_text_actions(self.chat_view)
        self.message_entry = scrolledtext.ScrolledText(tab, height=4, wrap=tk.WORD, font=("Segoe UI", 10))
        self.message_entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.message_entry.bind("<Return>", self._chat_enter)
        self.message_entry.bind("<Shift-Return>", lambda event: None)
        self._bind_text_actions(self.message_entry)
        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=1, sticky="ns", padx=(8, 0), pady=(8, 0))
        ttk.Button(buttons, text="Отправить", command=self._send_message).pack(fill="x")
        ttk.Button(buttons, text="Сохранить диалог", command=self._save_dialogue).pack(fill="x", pady=6)

    def _build_training_tab(self) -> None:
        tab = self._tab("Обучение")
        self.training_text = scrolledtext.ScrolledText(tab, height=18, wrap=tk.WORD)
        self.training_text.pack(fill="both", expand=True)
        self._bind_text_actions(self.training_text)
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="Запустить тестовое обучение", command=self._start_training).pack(side="left", padx=4)
        ttk.Button(controls, text="Пауза", command=lambda: logger.log("Пауза обучения запрошена")).pack(side="left", padx=4)
        ttk.Button(controls, text="Остановить", command=lambda: logger.log("Остановка обучения запрошена")).pack(side="left", padx=4)

    def _build_teacher_tab(self) -> None:
        tab = self._tab("Teacher")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew")
        self.teacher_topic = tk.StringVar(value="Transformer architecture")
        self.assistant_backend = tk.StringVar(value="transformers")
        self.assistant_model = tk.StringVar(value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        ttk.Label(form, text="Тема:").pack(side="left")
        ttk.Entry(form, textvariable=self.teacher_topic, width=32).pack(side="left", padx=6)
        ttk.Label(form, text="Assistant backend:").pack(side="left")
        ttk.Combobox(form, textvariable=self.assistant_backend, values=["transformers", "gguf", "ollama"], width=14).pack(side="left", padx=6)
        ttk.Entry(form, textvariable=self.assistant_model, width=38).pack(side="left", padx=6)
        ttk.Button(form, text="Сгенерировать задания", command=self._teacher_generate).pack(side="left", padx=4)
        ttk.Button(form, text="Проверить помощника", command=self._assistant_check).pack(side="left", padx=4)
        self.teacher_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.teacher_text.grid(row=1, column=0, sticky="nsew", pady=8)
        self._bind_text_actions(self.teacher_text)

    def _build_model_tab(self) -> None:
        tab = self._tab("Модели")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        columns = ("version", "stage", "created", "params", "size", "dataset", "loss", "ppl")
        self.model_tree = ttk.Treeview(tab, columns=columns, show="headings")
        headings = ["Версия", "Стадия", "Дата", "Параметры", "Размер", "Датасет", "Loss", "PPL"]
        for col, heading in zip(columns, headings):
            self.model_tree.heading(col, text=heading)
            self.model_tree.column(col, width=120, anchor="w")
        self.model_tree.grid(row=0, column=0, sticky="nsew")
        controls = ttk.Frame(tab)
        controls.grid(row=1, column=0, sticky="ew", pady=8)
        ttk.Button(controls, text="Загрузить checkpoint", command=self._load_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Продвинуть candidate", command=self._promote_candidate).pack(side="left", padx=4)
        ttk.Button(controls, text="Откат production", command=self._rollback_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Обновить", command=self._refresh_models).pack(side="left", padx=4)

    def _build_diagnostics_tab(self) -> None:
        tab = self._tab("Диагностика системы")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.diagnostic_tree = ttk.Treeview(tab, columns=("component", "status", "message"), show="headings")
        for col, heading, width in [("component", "Компонент", 220), ("status", "Статус", 100), ("message", "Описание", 700)]:
            self.diagnostic_tree.heading(col, text=heading)
            self.diagnostic_tree.column(col, width=width, anchor="w")
        self.diagnostic_tree.tag_configure("OK", foreground="#0a7f36")
        self.diagnostic_tree.tag_configure("WARNING", foreground="#a36b00")
        self.diagnostic_tree.tag_configure("ERROR", foreground="#b00020")
        self.diagnostic_tree.grid(row=0, column=0, sticky="nsew")
        ttk.Button(tab, text="Обновить диагностику", command=self._refresh_diagnostics).grid(row=1, column=0, sticky="w", pady=8)

    def _build_memory_tab(self) -> None:
        tab = self._tab("Память")
        self.memory_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.memory_text.pack(fill="both", expand=True)
        self._bind_text_actions(self.memory_text)

    def _build_datasets_tab(self) -> None:
        tab = self._tab("Датасеты")
        self.datasets_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.datasets_text.pack(fill="both", expand=True)
        self._bind_text_actions(self.datasets_text)
        ttk.Button(tab, text="Обновить список датасетов", command=self._refresh_datasets).pack(anchor="w", pady=8)

    def _build_metrics_tab(self) -> None:
        tab = self._tab("Метрики")
        self.metrics_canvas = tk.Canvas(tab, height=260, background="white")
        self.metrics_canvas.pack(fill="x")
        self.metrics_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.metrics_text.pack(fill="both", expand=True)
        self._bind_text_actions(self.metrics_text)

    def _build_logs_tab(self) -> None:
        tab = self._tab("Логи")
        self.log_view = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.log_view.pack(fill="both", expand=True)
        self._bind_text_actions(self.log_view)
        ttk.Button(tab, text="Очистить лог", command=self._clear_log).pack(anchor="w", pady=8)

    def _build_settings_tab(self) -> None:
        tab = self._tab("Настройки")
        text = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        text.pack(fill="both", expand=True)
        text.insert("end", json.dumps({"model": str(self.model_config), "root": str(ROOT_DIR)}, ensure_ascii=False, indent=2))
        self._bind_text_actions(text)

    def _show_initializer_if_needed(self) -> None:
        missing = self.diagnostics.missing_required_components()
        if not missing:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Первая инициализация AI Genesis")
        dialog.geometry("720x460")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(dialog, text="AI Genesis ещё не инициализирован", style="Title.TLabel").pack(anchor="w", padx=16, pady=(16, 6))
        description = scrolledtext.ScrolledText(dialog, height=14, wrap=tk.WORD)
        description.pack(fill="both", expand=True, padx=16, pady=8)
        description.insert("end", "Отсутствуют обязательные компоненты:\n\n")
        for item in missing:
            description.insert("end", f"• {item.name}: {item.message}\n")
        description.insert("end", "\nНажмите кнопку ниже, чтобы создать каталоги, базы данных, tokenizer, dataset, candidate/production модели и checkpoint.")
        description.configure(state="disabled")
        progress = tk.StringVar(value="Готово к запуску")
        ttk.Label(dialog, textvariable=progress).pack(fill="x", padx=16)
        button = ttk.Button(dialog, text="Инициализировать Genesis", command=lambda: self._run_initialization(dialog, button, progress))
        button.pack(pady=16)

    def _run_initialization(self, dialog: tk.Toplevel, button: ttk.Button, progress: tk.StringVar) -> None:
        button.configure(state="disabled")
        self.current_task.set("Инициализация")

        def worker() -> None:
            try:
                initializer = GenesisInitializer(progress=lambda message: self.after(0, progress.set, message))
                initializer.initialize()
                self.after(0, lambda: (dialog.destroy(), self._load_runtime_model(), self._refresh_all(), messagebox.showinfo("AI Genesis", "Инициализация успешно завершена")))
            except Exception as exc:  # noqa: BLE001
                logger.log(f"Ошибка инициализации: {exc}")
                self.after(0, lambda: (button.configure(state="normal"), messagebox.showerror("Ошибка инициализации", str(exc))))
            finally:
                self.after(0, self.current_task.set, "Ожидание")

        threading.Thread(target=worker, daemon=True).start()

    def _chat_enter(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:
            return None
        self._send_message()
        return "break"

    def _send_message(self) -> None:
        message = self.message_entry.get("1.0", "end").strip()
        if not message:
            return
        self.message_entry.delete("1.0", "end")
        self._append_chat("Вы", message)
        self.current_task.set("Генерация ответа")

        def worker() -> None:
            response = self.chat_engine.answer(message)
            self.after(0, lambda: (self._append_chat("Genesis", response), self.current_task.set("Ожидание"), self._refresh_memory()))

        threading.Thread(target=worker, daemon=True).start()

    def _append_chat(self, role: str, text: str) -> None:
        self.chat_view.configure(state="normal")
        self.chat_view.insert("end", f"\n{role}:\n{text}\n")
        self.chat_view.see("end")
        self.chat_view.configure(state="disabled")

    def _save_dialogue(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            Path(path).write_text(self.chat_view.get("1.0", "end"), encoding="utf-8")
            logger.log(f"Диалог сохранён: {path}")

    def _start_training(self) -> None:
        self.current_task.set("Тестовое обучение")
        self.training_text.insert("end", "Тестовое обучение доступно через мастер инициализации или CLI --train.\n")
        logger.log("Запрос тестового обучения из GUI")
        self.current_task.set("Ожидание")

    def _teacher_generate(self) -> None:
        tasks = self.teacher.curriculum(self.teacher_topic.get(), per_level=2)
        self.teacher_text.delete("1.0", "end")
        for task in tasks:
            self.teacher_text.insert("end", f"[{task.level}] {task.question}\nОжидается: {', '.join(task.expected_traits)}\n\n")
        logger.log(f"Teacher сгенерировал задания по теме {self.teacher_topic.get()}")

    def _assistant_check(self) -> None:
        config = AssistantModelConfig(self.assistant_backend.get(), self.assistant_model.get())
        assistant = LocalAssistantModel(config)
        status = "доступен" if assistant.available() else "не установлен backend/dependency"
        self.teacher_text.insert("end", f"\nKnowledge Assistant ({config.backend}:{config.model}) {status}.\n")

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt")])
        if path:
            self.chat_engine.load_model(Path(path))
            logger.log(f"Модель загружена в чат: {path}")
            self._refresh_models()

    def _promote_candidate(self) -> None:
        if self.model_manager.promote_candidate():
            self._load_runtime_model()
            self._refresh_models()

    def _rollback_model(self) -> None:
        if self.model_manager.rollback():
            self._load_runtime_model()
            self._refresh_models()

    def _load_runtime_model(self) -> None:
        try:
            model, _ = self.model_manager.load_production()
            self.chat_engine.model = model
            if self.model_config.tokenizer_path.exists():
                self.chat_engine.tokenizer = GenesisTokenizer(self.model_config.tokenizer_path)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"Production model not loaded: {exc}")

    def _refresh_all(self) -> None:
        self._refresh_header()
        self._refresh_chat_history()
        self._refresh_models()
        self._refresh_diagnostics()
        self._refresh_memory()
        self._refresh_datasets()
        self._refresh_metrics()
        self._refresh_logs()
        self.after(self.gui_config.refresh_ms, self._refresh_all)

    def _refresh_header(self) -> None:
        status = self.model_manager.current_status()
        snapshot = self.monitor.snapshot()
        self.model_name_var.set(str(status.get("model_name", self.model_config.model_name)))
        self.version_var.set(f"Версия: {status.get('version', 'MVP не инициализирован')}")
        params = int(status.get("parameters") or 0)
        self.params_var.set(f"Параметры: {params:,}" if params else "Параметры: ожидают инициализации")
        size = int(status.get("size_bytes") or 0)
        self.size_var.set(f"Размер: {self._format_size(size)}" if size else "Размер: ожидает инициализации")
        device = snapshot.gpu_name if snapshot.gpu_name not in {"cpu-only", "unavailable"} else "CPU"
        self.device_var.set(f"Устройство: {device}")
        self.memory_var.set(f"RAM {snapshot.ram_used_gb:.2f}GB/{snapshot.ram_percent:.0f}% | VRAM {snapshot.vram_used_mb or 0:.0f}/{snapshot.vram_total_mb or 0:.0f}MB | GPU {snapshot.gpu_percent if snapshot.gpu_percent is not None else 'n/a'}% | CPU {snapshot.cpu_percent:.0f}%")
        self.train_status_var.set(f"Обучение: {snapshot.training_status}")

    def _refresh_chat_history(self) -> None:
        if getattr(self, "_chat_loaded", False):
            return
        for role, content in self.memory.recent_dialogue(30):
            self._append_chat(role, content)
        self._chat_loaded = True

    def _refresh_models(self) -> None:
        for row in self.model_tree.get_children():
            self.model_tree.delete(row)
        for record in self.model_manager.history(100):
            payload = record.get("payload", {})
            if not isinstance(payload, dict) or "version" not in payload:
                continue
            self.model_tree.insert("", "end", values=(payload.get("version"), payload.get("stage"), payload.get("created_at"), payload.get("parameters"), self._format_size(int(payload.get("size_bytes") or 0)), payload.get("dataset"), payload.get("loss"), payload.get("perplexity")))

    def _refresh_diagnostics(self) -> None:
        for row in self.diagnostic_tree.get_children():
            self.diagnostic_tree.delete(row)
        for item in self.diagnostics.run():
            self.diagnostic_tree.insert("", "end", values=(item.name, item.status, item.message), tags=(item.status,))

    def _refresh_memory(self) -> None:
        self.memory_text.delete("1.0", "end")
        self.memory_text.insert("end", "Диалоги:\n")
        for role, content in self.memory.recent_dialogue(10):
            self.memory_text.insert("end", f"- {role}: {content}\n")
        self.memory_text.insert("end", "\nФакты:\n")
        for fact in self.memory.recent_facts(10):
            self.memory_text.insert("end", f"- {fact}\n")
        self.memory_text.insert("end", "\nСобытия обучения:\n")
        for event, metrics in self.memory.recent_learning_events(10):
            self.memory_text.insert("end", f"- {event}: {metrics}\n")

    def _refresh_datasets(self) -> None:
        self.datasets_text.delete("1.0", "end")
        for dataset in self.datasets.list_datasets():
            self.datasets_text.insert("end", f"{dataset.name}: docs={dataset.documents}, tokens={dataset.tokens}, size={self._format_size(dataset.size_bytes)}, quality={dataset.quality_score}, path={dataset.path}\n")

    def _refresh_metrics(self) -> None:
        metrics = self.metrics.recent(self.gui_config.metrics_points)
        self.metrics_text.delete("1.0", "end")
        for metric in metrics[-20:]:
            self.metrics_text.insert("end", f"run={metric.run_id} epoch={metric.epoch} step={metric.step} loss={metric.loss:.4f} ppl={metric.perplexity}\n")
        self._draw_loss_chart(metrics)

    def _draw_loss_chart(self, metrics) -> None:
        self.metrics_canvas.delete("all")
        if not metrics:
            self.metrics_canvas.create_text(20, 20, anchor="w", text="Нет метрик обучения")
            return
        width = max(self.metrics_canvas.winfo_width(), 600)
        height = 240
        losses = [metric.loss for metric in metrics]
        lo, hi = min(losses), max(losses)
        span = max(hi - lo, 1e-6)
        points = []
        for index, loss in enumerate(losses):
            x = 20 + index * (width - 40) / max(len(losses) - 1, 1)
            y = height - 20 - ((loss - lo) / span) * (height - 40)
            points.extend([x, y])
        if len(points) >= 4:
            self.metrics_canvas.create_line(points, fill="#2563eb", width=2)

    def _refresh_logs(self) -> None:
        self.log_view.delete("1.0", "end")
        self.log_view.insert("end", "\n".join(logger.snapshot()))
        self.log_view.see("end")

    def _clear_log(self) -> None:
        logger.clear()
        self._refresh_logs()

    def _format_size(self, size: int) -> str:
        if size <= 0:
            return "0 байт"
        units = ["байт", "KB", "MB", "GB"]
        value = float(size)
        unit = 0
        while value >= 1024 and unit < len(units) - 1:
            value /= 1024
            unit += 1
        return f"{value:.1f} {units[unit]}"


def run_gui() -> None:
    app = GenesisGUI()
    app.mainloop()
