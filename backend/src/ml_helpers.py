from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_manager import get_dataset, get_model_prep_plan, get_schema_profile
from src.preprocessing import ModelPrepPlanError, infer_positive_label, prepare_model_inputs
from src.result_types import artifact_registry, build_artifact
from src.settings import SETTINGS


class MLHelperError(ValueError):
    pass


class BaselineMLService:
    def __init__(self, *, dataset_id: str):
        self.dataset_id = dataset_id

    def logistic_fit(
        self,
        target: str,
        *,
        features: list[str] | None = None,
        test_size: float | None = None,
        positive_label: Any = None,
    ) -> dict[str, Any]:
        dataset = get_dataset(self.dataset_id)
        profile_artifact = get_schema_profile(self.dataset_id)
        plan = get_model_prep_plan(self.dataset_id, target=target, features=features)
        bundle = self._prepare_inputs(dataset.raw_df, profile_artifact, plan, target=target, features=features)

        y = bundle.y.copy(deep=True)
        unique_labels = sorted(y.dropna().astype(str).unique().tolist())
        if len(unique_labels) != 2:
            raise MLHelperError(f"logistic regression 仅支持二分类目标，当前标签数为 {len(unique_labels)}。")

        inferred = infer_positive_label(y, explicit_label=positive_label)
        if inferred["source"] == "ambiguous" or inferred["positive_label"] is None:
            raise MLHelperError("目标正类无法自动推断，请显式提供 positive_label。")

        y_binary = self._map_positive(y, inferred["positive_label"])
        if y_binary.nunique() < 2:
            raise MLHelperError("目标列映射后仅剩单一类别，无法训练二分类模型。")

        test_size_value = self._resolve_test_size(test_size)
        self._validate_stratified_split(y_binary, test_size_value)
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                bundle.x,
                y_binary,
                test_size=test_size_value,
                random_state=SETTINGS.ml_random_state,
                stratify=y_binary,
            )
        except ValueError as exc:
            raise MLHelperError("目标列类别分布不满足当前测试集划分要求，请增加样本或调整 test_size。") from exc
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise MLHelperError("训练集或测试集类别不足，无法稳定评估模型。")

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(bundle.preprocessor)),
                ("model", LogisticRegression(random_state=SETTINGS.ml_random_state, max_iter=500)),
            ]
        )
        model_pipeline.fit(x_train, y_train)
        y_pred = model_pipeline.predict(x_test)
        y_prob = model_pipeline.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": self._safe_float(accuracy_score(y_test, y_pred)),
            "precision": self._safe_float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": self._safe_float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": self._safe_float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": self._safe_float(roc_auc_score(y_test, y_prob)),
        }
        coefficients = self._extract_coefficients(model_pipeline)

        warnings = list(bundle.warnings)
        if inferred["warning"]:
            warnings.append(str(inferred["warning"]))
        if not coefficients:
            warnings.append("未能稳定提取模型系数，feature_importance 可能受限。")

        payload = {
            "model_type": "logistic_regression",
            "target": bundle.target,
            "target_type": "binary_classification",
            "positive_label": inferred["positive_label"],
            "positive_label_source": inferred["source"],
            "features_used": bundle.features_used,
            "excluded_columns": bundle.excluded_columns,
            "train_rows": int(len(x_train.index)),
            "test_rows": int(len(x_test.index)),
            "class_balance": self._class_balance(y),
            "metrics": metrics,
            "coefficient_items": coefficients,
            "prep_summary": bundle.prep_summary,
        }
        artifact = build_artifact(
            artifact_type="model_result",
            dataset_id=self.dataset_id,
            payload=payload,
            warnings=warnings,
        )
        return artifact_registry.register(self.dataset_id, artifact)

    def linear_regression_fit(
        self,
        target: str,
        *,
        features: list[str] | None = None,
        test_size: float | None = None,
    ) -> dict[str, Any]:
        dataset = get_dataset(self.dataset_id)
        profile_artifact = get_schema_profile(self.dataset_id)
        plan = get_model_prep_plan(self.dataset_id, target=target, features=features)
        bundle = self._prepare_inputs(dataset.raw_df, profile_artifact, plan, target=target, features=features)

        y_numeric = pd.to_numeric(bundle.y, errors="coerce")
        valid_mask = y_numeric.notna()
        x = bundle.x.loc[valid_mask].copy(deep=True)
        y = y_numeric.loc[valid_mask]
        if len(y.index) < 20:
            raise MLHelperError("有效样本不足，建议至少 20 行再训练线性回归模型。")

        test_size_value = self._resolve_test_size(test_size)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size_value,
            random_state=SETTINGS.ml_random_state,
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(bundle.preprocessor)),
                ("model", LinearRegression()),
            ]
        )
        model_pipeline.fit(x_train, y_train)
        y_pred = model_pipeline.predict(x_test)

        metrics = {
            "rmse": self._safe_float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": self._safe_float(mean_absolute_error(y_test, y_pred)),
            "r2": self._safe_float(r2_score(y_test, y_pred)),
        }
        coefficients = self._extract_coefficients(model_pipeline)

        warnings = list(bundle.warnings)
        if len(y.index) != len(bundle.y.index):
            warnings.append(f"目标列转换为数值后丢弃了 {int(len(bundle.y.index) - len(y.index))} 行。")
        if not coefficients:
            warnings.append("未能稳定提取线性回归系数。")

        payload = {
            "model_type": "linear_regression",
            "target": bundle.target,
            "target_type": "numeric_regression",
            "features_used": bundle.features_used,
            "excluded_columns": bundle.excluded_columns,
            "train_rows": int(len(x_train.index)),
            "test_rows": int(len(x_test.index)),
            "metrics": metrics,
            "coefficient_items": coefficients,
            "prep_summary": bundle.prep_summary,
        }
        artifact = build_artifact(
            artifact_type="model_result",
            dataset_id=self.dataset_id,
            payload=payload,
            warnings=warnings,
        )
        return artifact_registry.register(self.dataset_id, artifact)

    def metrics(self, *, model_artifact_id: str | None = None) -> dict[str, Any]:
        model_artifact = self._resolve_model_artifact(model_artifact_id=model_artifact_id)
        payload = {
            "source_model_artifact_id": model_artifact["artifact_id"],
            "model_type": model_artifact.get("model_type"),
            "metrics": model_artifact.get("metrics", {}),
        }
        artifact = build_artifact(
            artifact_type="metrics_result",
            dataset_id=self.dataset_id,
            payload=payload,
            warnings=list(model_artifact.get("warnings", [])),
        )
        return artifact_registry.register(self.dataset_id, artifact)

    def feature_importance(self, *, model_artifact_id: str | None = None, top_k: int = 10) -> dict[str, Any]:
        if top_k <= 0:
            raise MLHelperError("top_k 必须为正整数。")
        model_artifact = self._resolve_model_artifact(model_artifact_id=model_artifact_id)
        coefficient_items = model_artifact.get("coefficient_items")
        warnings: list[str] = []
        if not isinstance(coefficient_items, list):
            coefficient_items = []
        if not coefficient_items:
            warnings.append("当前模型未提供可解释的系数特征重要性。")
        sorted_items = sorted(
            [item for item in coefficient_items if isinstance(item, dict)],
            key=lambda item: float(item.get("abs_importance", 0.0)),
            reverse=True,
        )
        limit = min(top_k, SETTINGS.ml_max_importance_items)
        items = sorted_items[:limit]
        payload = {
            "source_model_artifact_id": model_artifact["artifact_id"],
            "model_type": model_artifact.get("model_type"),
            "items": items,
            "top_k": limit,
        }
        artifact = build_artifact(
            artifact_type="feature_importance_result",
            dataset_id=self.dataset_id,
            payload=payload,
            warnings=warnings,
        )
        return artifact_registry.register(self.dataset_id, artifact)

    def latest(self, artifact_type: str | None = None) -> dict[str, Any]:
        artifact = artifact_registry.get_latest(self.dataset_id, artifact_type=artifact_type or "model_result")
        if artifact is None:
            raise MLHelperError("当前还没有可复用的模型结果，请先训练 baseline 模型。")
        return artifact

    def _prepare_inputs(
        self,
        df: pd.DataFrame,
        profile_artifact: dict[str, Any],
        plan: dict[str, Any],
        *,
        target: str | None,
        features: list[str] | None,
    ) -> Any:
        try:
            return prepare_model_inputs(
                df,
                profile_artifact,
                plan,
                target=target,
                features=features,
            )
        except ModelPrepPlanError as exc:
            raise MLHelperError(str(exc)) from exc

    def _resolve_model_artifact(self, *, model_artifact_id: str | None) -> dict[str, Any]:
        artifact: dict[str, Any] | None = None
        if model_artifact_id:
            artifact = artifact_registry.get_by_artifact_id(model_artifact_id)
            if artifact is None:
                raise MLHelperError("指定的模型 artifact 不存在。")
            if artifact.get("dataset_id") != self.dataset_id:
                raise MLHelperError("指定的模型 artifact 不属于当前数据集。")
        else:
            artifact = artifact_registry.get_latest(self.dataset_id, artifact_type="model_result")
            if artifact is None:
                raise MLHelperError("当前没有可用模型结果，请先训练 baseline 模型。")

        if artifact.get("artifact_type") != "model_result":
            raise MLHelperError("指定 artifact 不是 model_result。")
        return artifact

    def _resolve_test_size(self, test_size: float | None) -> float:
        value = SETTINGS.ml_default_test_size if test_size is None else float(test_size)
        if value <= 0.05 or value >= 0.5:
            raise MLHelperError("test_size 需在 (0.05, 0.5) 区间内。")
        return value

    def _validate_stratified_split(self, y_binary: pd.Series, test_size: float) -> None:
        total_rows = int(len(y_binary.index))
        if total_rows < 10:
            raise MLHelperError("样本量过小，无法进行分层训练/测试划分。")

        counts = y_binary.value_counts(dropna=False)
        if len(counts.index) != 2:
            raise MLHelperError("目标列需要恰好两类才能进行分层二分类训练。")

        minority = int(counts.min())
        test_rows = int(round(total_rows * test_size))
        train_rows = total_rows - test_rows
        if test_rows < 2 or train_rows < 2:
            raise MLHelperError("当前 test_size 导致训练集或测试集过小，无法稳定训练。")

        # Stratified split requires each class to appear in both train and test.
        # Conservative rule: minority class should have at least 2 samples.
        if minority < 2:
            raise MLHelperError("目标列类别分布不平衡且少数类样本不足，无法进行分层划分。")

    def _extract_coefficients(self, model_pipeline: Pipeline) -> list[dict[str, Any]]:
        model = model_pipeline.named_steps.get("model")
        preprocessor = model_pipeline.named_steps.get("preprocessor")
        if model is None or preprocessor is None or not hasattr(model, "coef_"):
            return []
        try:
            names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            names = []
        coef = model.coef_
        if isinstance(coef, np.ndarray) and coef.ndim > 1:
            coef = coef[0]
        if not isinstance(coef, np.ndarray):
            return []
        if not names or len(names) != len(coef):
            names = [f"f_{idx}" for idx in range(len(coef))]
        items = []
        for idx, value in enumerate(coef):
            coef_value = float(value)
            items.append(
                {
                    "feature": str(names[idx]),
                    "coefficient": round(coef_value, 6),
                    "abs_importance": round(abs(coef_value), 6),
                }
            )
        items.sort(key=lambda item: item["abs_importance"], reverse=True)
        return items[: SETTINGS.ml_max_importance_items]

    def _class_balance(self, series: pd.Series) -> dict[str, int]:
        counts = series.astype(str).value_counts(dropna=False)
        return {str(index): int(value) for index, value in counts.items()}

    def _map_positive(self, series: pd.Series, positive_label: Any) -> pd.Series:
        if isinstance(positive_label, bool):
            return series.fillna(False).astype(bool).eq(bool(positive_label)).astype(int)
        if isinstance(positive_label, (int, float, np.integer, np.floating)):
            return pd.to_numeric(series, errors="coerce").eq(float(positive_label)).astype(int)
        return series.astype(str).str.strip().str.lower().eq(str(positive_label).strip().lower()).astype(int)

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return round(float(value), 6)
