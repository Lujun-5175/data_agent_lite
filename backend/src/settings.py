from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    preview_row_count: int = _get_int("PREVIEW_ROW_COUNT", 10)
    artifact_ttl_seconds: int = _get_int("ARTIFACT_TTL_SECONDS", 24 * 60 * 60)
    max_upload_size_bytes: int = _get_int("MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024)
    upload_chunk_size: int = _get_int("UPLOAD_CHUNK_SIZE", 1024 * 1024)
    artifact_cleanup_interval_seconds: int = _get_int("ARTIFACT_CLEANUP_INTERVAL_SECONDS", 60 * 60)
    max_auto_numeric_columns: int = _get_int("MAX_AUTO_NUMERIC_COLUMNS", 20)
    max_auto_categorical_columns: int = _get_int("MAX_AUTO_CATEGORICAL_COLUMNS", 20)
    max_corr_columns: int = _get_int("MAX_CORR_COLUMNS", 12)
    max_top_pairs: int = _get_int("MAX_TOP_PAIRS", 20)
    max_group_rows: int = _get_int("MAX_GROUP_ROWS", 500)
    default_group_top_n: int = _get_int("DEFAULT_GROUP_TOP_N", 10)
    profile_sample_values_count: int = _get_int("PROFILE_SAMPLE_VALUES_COUNT", 5)
    profile_datetime_parse_ratio_threshold: float = float(os.getenv("PROFILE_DATETIME_PARSE_RATIO_THRESHOLD", "0.8"))
    profile_identifier_unique_ratio_threshold: float = float(
        os.getenv("PROFILE_IDENTIFIER_UNIQUE_RATIO_THRESHOLD", "0.98")
    )
    profile_text_avg_length_threshold: float = float(os.getenv("PROFILE_TEXT_AVG_LENGTH_THRESHOLD", "24"))
    profile_high_cardinality_ratio_threshold: float = float(
        os.getenv("PROFILE_HIGH_CARDINALITY_RATIO_THRESHOLD", "0.5")
    )
    profile_high_missing_ratio_threshold: float = float(os.getenv("PROFILE_HIGH_MISSING_RATIO_THRESHOLD", "0.3"))
    model_prep_max_target_candidates: int = _get_int("MODEL_PREP_MAX_TARGET_CANDIDATES", 5)
    ml_max_training_rows: int = _get_int("ML_MAX_TRAINING_ROWS", 10000)
    ml_max_feature_count: int = _get_int("ML_MAX_FEATURE_COUNT", 30)
    ml_default_test_size: float = float(os.getenv("ML_DEFAULT_TEST_SIZE", "0.2"))
    ml_random_state: int = _get_int("ML_RANDOM_STATE", 42)
    ml_max_importance_items: int = _get_int("ML_MAX_IMPORTANCE_ITEMS", 20)
    ml_max_ohe_categories: int = _get_int("ML_MAX_OHE_CATEGORIES", 30)
    routing_dataset_required_threshold: float = float(os.getenv("ROUTING_DATASET_REQUIRED_THRESHOLD", "3.0"))
    routing_stats_intent_threshold: float = float(os.getenv("ROUTING_STATS_INTENT_THRESHOLD", "2.8"))
    routing_ml_intent_threshold: float = float(os.getenv("ROUTING_ML_INTENT_THRESHOLD", "2.8"))
    positive_label_hints: tuple[str, ...] = ("yes", "true", "1", "positive", "churn", "是", "阳性")
    negative_label_hints: tuple[str, ...] = ("no", "false", "0", "negative", "非流失", "否", "阴性")


SETTINGS = AppSettings()
