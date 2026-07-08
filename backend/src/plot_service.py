"""
Plotting service — PlotHelperAPI. Extracted from tools.py.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.safe_executor import SafeExecutionError


class PlotHelperAPI:
    """Safe plotting API exposed to the LLM's Python execution context."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def hist(self, column: str, bins: int = 30, title: str | None = None):
        if column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {column}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data=self._df, x=column, bins=bins, ax=ax, color="#4F86C6")
        ax.set_title(title or f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        return fig

    def bar(self, x: str, y: str | None = None, title: str | None = None, top_n: int = 10):
        if x not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {x}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if y and y in self._df.columns:
            summary = self._df.groupby(x, dropna=False)[y].mean().reset_index(name=y)
            sns.barplot(data=summary.head(top_n), x=x, y=y, ax=ax, color="#4F86C6")
            ax.set_ylabel(y)
        else:
            counts = self._df[x].value_counts(dropna=False).head(top_n).reset_index()
            counts.columns = [x, "count"]
            sns.barplot(data=counts, x=x, y="count", ax=ax, color="#4F86C6")
            ax.set_ylabel("count")
        ax.set_title(title or f"{x} bar chart")
        ax.set_xlabel(x)
        fig.tight_layout()
        return fig

    def line(self, x: str, y: str, title: str | None = None):
        if x not in self._df.columns or y not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {x} / {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.lineplot(data=self._df, x=x, y=y, ax=ax, color="#4F86C6")
        ax.set_title(title or f"{x} vs {y}")
        fig.tight_layout()
        return fig

    def scatter(self, x: str, y: str, hue: str | None = None, title: str | None = None):
        if x not in self._df.columns or y not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {x} / {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=self._df, x=x, y=y, hue=hue if hue in self._df.columns else None, ax=ax)
        ax.set_title(title or f"{x} vs {y}")
        fig.tight_layout()
        return fig

    def box(self, y: str, x: str | None = None, title: str | None = None):
        if y not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=self._df, x=x if x in self._df.columns else None, y=y, ax=ax)
        ax.set_title(title or f"{y} box plot")
        fig.tight_layout()
        return fig

    def heatmap_corr(self, columns: list[str] | None = None, title: str | None = None):
        if columns:
            target_df = self._df.loc[:, [c for c in columns if c in self._df.columns]]
        else:
            target_df = self._df.select_dtypes(include=[np.number])
        if target_df.empty:
            raise SafeExecutionError("No numeric columns available for correlation heatmap.")
        corr_df = target_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, cmap="Blues", annot=True, fmt=".2f", ax=ax)
        ax.set_title(title or "Correlation heatmap")
        fig.tight_layout()
        return fig
