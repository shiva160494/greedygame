from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from . import OUTCOMES
from .analysis import frequencies, streak_summary
from .backtest import walk_forward
from .engine import PredictionEngine
from .storage import HistoryStore


class GreedyGameApp(tk.Tk):
    """Small local desktop interface for recording and analysing outcomes."""

    def __init__(self, data_file: str | Path):
        super().__init__()
        self.store = HistoryStore(data_file)
        self.weights_file = Path(data_file).parent / "model_weights.json"
        self.title("Greedy Game Analyzer")
        self.geometry("980x680")
        self.minsize(820, 560)
        self.configure(bg="#101827")
        self._style()
        self._build()
        self.refresh()

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background="#172033", foreground="#ecf2ff", fieldbackground="#172033", rowheight=28)
        style.configure("Treeview.Heading", background="#243149", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#2563eb")])

    def _build(self) -> None:
        header = tk.Frame(self, bg="#101827")
        header.pack(fill="x", padx=24, pady=(20, 12))
        tk.Label(header, text="Greedy Game Analyzer", font=("Segoe UI", 22, "bold"), fg="#f8fafc", bg="#101827").pack(anchor="w")
        tk.Label(header, text="Historical analysis only — not a guarantee of future game results.", font=("Segoe UI", 10), fg="#9fb0cc", bg="#101827").pack(anchor="w", pady=(2, 0))

        self.status = tk.StringVar()
        tk.Label(header, textvariable=self.status, font=("Segoe UI", 10), fg="#67e8f9", bg="#101827").pack(anchor="e")

        controls = tk.Frame(self, bg="#172033", highlightbackground="#2d3c58", highlightthickness=1)
        controls.pack(fill="x", padx=24, pady=8)
        tk.Label(controls, text="Add the latest outcome", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#172033").pack(anchor="w", padx=16, pady=(12, 8))
        button_row = tk.Frame(controls, bg="#172033")
        button_row.pack(fill="x", padx=12, pady=(0, 14))
        for outcome in OUTCOMES:
            tk.Button(button_row, text=outcome, command=lambda value=outcome: self.add(value), relief="flat", bd=0,
                      font=("Segoe UI", 10, "bold"), bg="#2563eb", fg="white", activebackground="#1d4ed8", activeforeground="white",
                      padx=12, pady=8, cursor="hand2").pack(side="left", padx=4)

        actions = tk.Frame(self, bg="#101827")
        actions.pack(fill="x", padx=24, pady=8)
        for text, callback in [("Refresh", self.refresh), ("Saved History", self.show_history), ("Run Backtest", self.backtest), ("Show All Statistics", self.show_stats)]:
            tk.Button(actions, text=text, command=callback, relief="flat", bg="#334155", fg="white", activebackground="#475569",
                      padx=12, pady=7, cursor="hand2").pack(side="left", padx=(0, 8))

        content = tk.Frame(self, bg="#101827")
        content.pack(fill="both", expand=True, padx=24, pady=(4, 24))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        left = tk.Frame(content, bg="#172033", highlightbackground="#2d3c58", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(left, text="Next Top 5 — Historical Model", font=("Segoe UI", 13, "bold"), fg="#f8fafc", bg="#172033").pack(anchor="w", padx=16, pady=(14, 2))
        self.evidence = tk.StringVar()
        tk.Label(left, textvariable=self.evidence, font=("Segoe UI", 9), fg="#9fb0cc", bg="#172033").pack(anchor="w", padx=16, pady=(0, 10))
        self.ranking = ttk.Treeview(left, columns=("rank", "outcome", "score"), show="headings", height=8)
        for column, heading, width in [("rank", "Rank", 70), ("outcome", "Next outcome", 180), ("score", "Model score", 150)]:
            self.ranking.heading(column, text=heading)
            self.ranking.column(column, width=width, anchor="center")
        self.ranking.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        right = tk.Frame(content, bg="#172033", highlightbackground="#2d3c58", highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(right, text="Outcome frequency", font=("Segoe UI", 13, "bold"), fg="#f8fafc", bg="#172033").pack(anchor="w", padx=16, pady=(14, 10))
        self.frequency = ttk.Treeview(right, columns=("outcome", "count", "percent"), show="headings", height=8)
        for column, heading, width in [("outcome", "Outcome", 125), ("count", "Count", 70), ("percent", "%", 75)]:
            self.frequency.heading(column, text=heading)
            self.frequency.column(column, width=width, anchor="center")
        self.frequency.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    @staticmethod
    def _clear(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def add(self, outcome: str) -> None:
        self.store.append([outcome])
        self.refresh()

    def refresh(self) -> None:
        history = self.store.load()
        current = streak_summary(history)
        self.status.set(f"Saved outcomes: {len(history)}  • Current streak: {current['current_outcome'] or '—'} ×{current['current_length']}")
        prediction = PredictionEngine(weights_file=self.weights_file).predict(history)
        self._clear(self.ranking)
        for rank, (outcome, score) in enumerate(prediction["ranking"][:5], 1):
            self.ranking.insert("", "end", values=(rank, outcome, f"{score:.2%}"))
        self.evidence.set(f"Based on {prediction['support']}/4 pattern models • scores are historical, not guarantees")
        self._clear(self.frequency)
        for row in sorted(frequencies(history), key=lambda value: (-value["count"], value["outcome"])):
            self.frequency.insert("", "end", values=(row["outcome"], row["count"], f"{row['percent']:.2f}%"))

    def backtest(self) -> None:
        history = self.store.load()
        if len(history) < 31:
            messagebox.showinfo("Need more history", "Add at least 31 outcomes before running a backtest.")
            return
        result = walk_forward(history)
        messagebox.showinfo("Walk-forward backtest", f"Predictions tested: {result['predictions']}\n\nTop-1 accuracy: {result['top1_accuracy']}%\nTop-3 accuracy: {result['top3_accuracy']}%\nTop-5 accuracy: {result['top5_accuracy']}%\n\nUniform 8-outcome Top-1 baseline: 12.5%")

    def show_stats(self) -> None:
        history = self.store.load()
        current = streak_summary(history)
        details = "\n".join(f"{item}: longest streak {info['longest']}" for item, info in current["by_outcome"].items())
        messagebox.showinfo("Streak statistics", details or "No outcomes recorded.")

    def show_history(self) -> None:
        """Display every saved spin in a scrollable separate window."""
        history = self.store.load()
        window = tk.Toplevel(self)
        window.title(f"Saved Outcomes ({len(history)})")
        window.geometry("420x560")
        window.configure(bg="#101827")
        tk.Label(window, text=f"Saved Outcomes — {len(history)} total", font=("Segoe UI", 14, "bold"), fg="#f8fafc", bg="#101827").pack(anchor="w", padx=16, pady=16)
        frame = tk.Frame(window, bg="#101827")
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        table = ttk.Treeview(frame, columns=("spin", "outcome"), show="headings")
        table.heading("spin", text="Spin")
        table.heading("outcome", text="Saved outcome")
        table.column("spin", width=100, anchor="center")
        table.column("outcome", width=220, anchor="center")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for spin, outcome in enumerate(history, 1):
            table.insert("", "end", values=(spin, outcome))
        if history:
            table.yview_moveto(1.0)


def launch(data_file: str | Path) -> None:
    GreedyGameApp(data_file).mainloop()
