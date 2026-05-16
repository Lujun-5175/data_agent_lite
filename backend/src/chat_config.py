from __future__ import annotations

import re

FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "解释",
    "刚才",
    "之前",
    "上一个",
    "上个",
    "这个结果",
    "这结果",
    "上面的结果",
    "继续",
    "follow up",
    "follow-up",
    "再次",
    "again",
    "reuse",
)

CHART_MARKERS: tuple[str, ...] = (
    "画图",
    "绘图",
    "生成图",
    "生成一个图",
    "图表",
    "柱状图",
    "折线图",
    "散点图",
    "直方图",
    "热力图",
    "可视化",
    "chart",
    "plot",
    "visualize",
)

DATASET_OVERVIEW_MARKERS: tuple[str, ...] = (
    "讲解数据集",
    "介绍数据集",
    "解释数据集",
    "数据集概览",
    "数据集说明",
    "看看数据集",
    "了解数据集",
    "describe dataset",
    "explain dataset",
    "summarize dataset",
    "dataset overview",
)

DATASET_OVERVIEW_ACTION_TERMS: tuple[str, ...] = (
    "讲解",
    "介绍",
    "解释",
    "看看",
    "了解",
    "概览",
    "说明",
    "总结",
    "分析一下这个数据集",
    "describe",
    "explain",
    "summarize",
    "overview",
    "walk through",
)

DATASET_OVERVIEW_SUBJECT_TERMS: tuple[str, ...] = (
    "数据集",
    "这份数据",
    "这个数据集",
    "这个表",
    "这张表",
    "这份表",
    "dataset",
    "table",
    "csv",
)

ML_TRAINING_TERMS: tuple[str, ...] = (
    "train a model",
    "train a logistic regression model",
    "train a linear regression model",
    "train model",
    "build a model",
    "build model",
    "fit model",
    "baseline model",
    "classifier",
    "classification model",
    "logistic regression",
    "linear regression",
    "predict",
    "prediction",
    "forecast",
    "训练模型",
    "训练一个模型",
    "训练一个 baseline",
    "分类器",
    "分类模型",
    "逻辑回归",
    "线性回归",
    "预测",
    "预测一下",
)

ML_METRICS_TERMS: tuple[str, ...] = (
    "model metrics",
    "metrics",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "roc auc",
    "模型指标",
    "准确率",
    "精确率",
    "召回率",
)

ML_IMPORTANCE_TERMS: tuple[str, ...] = (
    "feature importance",
    "coefficients",
    "coefficient",
    "特征重要性",
    "重要特征",
    "系数",
)

DATASET_REQUIRED_WEIGHTS: dict[str, float] = {
    "数据集": 1.6,
    "数据": 1.2,
    "这份数据": 1.6,
    "当前数据": 1.6,
    "csv": 1.4,
    "上传": 1.2,
    "分析": 1.0,
    "相关性": 1.4,
    "分组": 1.2,
    "检验": 1.5,
    "group by": 1.3,
    "correlation": 1.4,
    "t-test": 1.5,
    "chi-square": 1.5,
}

STATS_INTENT_WEIGHTS: dict[str, float] = {
    "describe": 1.3,
    "summary": 1.2,
    "描述统计": 1.6,
    "分组汇总": 1.6,
    "group by": 1.4,
    "correlation": 1.4,
    "相关性": 1.4,
    "t-test": 1.8,
    "t检验": 1.8,
    "chi-square": 1.8,
    "卡方": 1.8,
    "anova": 1.8,
    "显著性": 1.6,
}

ML_INTENT_WEIGHTS: dict[str, float] = {
    "train model": 2.0,
    "baseline model": 1.8,
    "predict": 1.8,
    "prediction": 1.2,
    "classification": 1.8,
    "regression": 1.8,
    "logistic regression": 2.2,
    "linear regression": 2.2,
    "evaluate model": 1.8,
    "accuracy": 0.8,
    "feature importance": 1.8,
    "训练模型": 2.0,
    "预测": 1.8,
    "预测一下": 1.2,
    "分类": 1.6,
    "回归": 1.6,
    "逻辑回归": 2.2,
    "线性回归": 2.2,
    "模型评估": 1.8,
    "特征重要性": 1.8,
    "重要特征": 1.0,
}

EXPLICIT_ML_WEIGHTS: dict[str, float] = {
    "train a model": 2.4,
    "train a logistic regression model": 3.2,
    "train a linear regression model": 3.2,
    "train model": 2.4,
    "build a model": 2.4,
    "build model": 2.4,
    "try a simple model": 3.0,
    "simple model": 2.6,
    "try a model": 2.2,
    "fit model": 2.4,
    "baseline model": 2.2,
    "predict": 2.0,
    "prediction": 1.8,
    "classify": 2.2,
    "classifier": 2.2,
    "classification model": 2.2,
    "logistic regression": 2.6,
    "linear regression": 2.6,
    "evaluate model": 2.2,
    "model metrics": 2.6,
    "metrics": 2.0,
    "accuracy": 1.2,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "auc": 1.0,
    "roc auc": 1.2,
    "feature importance": 2.8,
    "coefficients": 1.6,
    "coefficient": 1.6,
    "train一个模型": 2.4,
    "训练一个模型": 2.4,
    "训练模型": 2.4,
    "训练一个 baseline": 2.2,
    "预测": 2.0,
    "预测一下": 2.0,
    "分类器": 2.2,
    "分类模型": 2.2,
    "逻辑回归": 2.6,
    "线性回归": 2.6,
    "模型评估": 2.2,
    "模型指标": 2.6,
    "准确率": 1.2,
    "精确率": 1.0,
    "召回率": 1.0,
    "f1分数": 1.0,
    "特征重要性": 2.8,
    "重要特征": 2.0,
}

EXPLORATORY_ANALYSIS_TERMS: dict[str, float] = {
    "analyze": 1.3,
    "analysis": 1.2,
    "explore": 1.2,
    "exploration": 1.2,
    "look at": 1.1,
    "compare": 1.2,
    "comparison": 1.1,
    "factors": 1.2,
    "drivers": 1.2,
    "influence": 1.1,
    "relationship": 1.1,
    "distribution": 1.0,
    "trend": 1.0,
    "why": 0.8,
    "reason": 0.8,
    "summary": 1.0,
    "describe": 1.0,
    "概括": 1.0,
    "分析": 1.2,
    "探索": 1.2,
    "看看": 0.8,
    "比较": 1.2,
    "因素": 1.2,
    "驱动": 1.2,
    "影响": 1.1,
    "关系": 1.1,
    "分布": 1.0,
    "趋势": 1.0,
    "原因": 0.8,
}

DELIVERABLE_TERM_MAP: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "概括", "总结", "overview", "describe", "分析"),
    "metrics": ("metrics", "model metrics", "模型指标", "accuracy", "precision", "recall", "f1", "auc", "roc auc"),
    "feature_importance": ("feature importance", "特征重要性", "重要特征", "coefficients", "coefficient", "系数"),
    "chart": CHART_MARKERS,
    "table": ("table", "表格", "明细", "top", "top n", "前", "列表"),
    "prediction": ("predict", "prediction", "预测", "预测一下", "forecast"),
    "explanation": FOLLOW_UP_MARKERS,
}

ANALYSIS_OPERATION_TERMS: dict[str, float] = {
    "group": 0.4,
    "compare": 0.4,
    "test": 0.5,
    "association": 0.5,
    "significance": 0.5,
    "分组": 0.4,
    "比较": 0.4,
    "检验": 0.5,
    "关联": 0.5,
    "显著": 0.5,
}


def normalize_message_text(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def contains_any_term(message: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in message for term in terms)


def looks_like_dataset_overview_fallback(message: str) -> bool:
    normalized = normalize_message_text(message)
    if not normalized:
        return False
    if contains_any_term(normalized, DATASET_OVERVIEW_MARKERS):
        return True
    return contains_any_term(normalized, DATASET_OVERVIEW_ACTION_TERMS) and contains_any_term(
        normalized,
        DATASET_OVERVIEW_SUBJECT_TERMS,
    )
