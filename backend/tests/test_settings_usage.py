from __future__ import annotations

from src import data_manager, server, tools
from src.settings import SETTINGS


def test_settings_provide_core_thresholds():
    assert SETTINGS.max_auto_numeric_columns > 0
    assert SETTINGS.max_auto_categorical_columns > 0
    assert SETTINGS.max_corr_columns > 1
    assert SETTINGS.max_group_rows > 0
    assert SETTINGS.max_upload_size_bytes > 0
    assert SETTINGS.profile_sample_values_count > 0
    assert SETTINGS.profile_datetime_parse_ratio_threshold > 0
    assert SETTINGS.profile_identifier_unique_ratio_threshold > 0
    assert SETTINGS.profile_text_avg_length_threshold > 0
    assert SETTINGS.ml_max_training_rows > 0
    assert SETTINGS.ml_max_feature_count > 0
    assert 0 < SETTINGS.ml_default_test_size < 0.5
    assert SETTINGS.ml_max_importance_items > 0


def test_consumers_use_settings_values():
    assert data_manager.PREVIEW_ROW_COUNT == SETTINGS.preview_row_count
    assert data_manager.ARTIFACT_TTL_SECONDS == SETTINGS.artifact_ttl_seconds
    assert server.MAX_UPLOAD_SIZE_BYTES == SETTINGS.max_upload_size_bytes
    assert server.UPLOAD_CHUNK_SIZE == SETTINGS.upload_chunk_size
    assert server.ARTIFACT_CLEANUP_INTERVAL_SECONDS == SETTINGS.artifact_cleanup_interval_seconds
    assert tools.StatsHelperAPI.MAX_AUTO_NUMERIC_COLUMNS == SETTINGS.max_auto_numeric_columns
    assert tools.StatsHelperAPI.MAX_AUTO_CATEGORICAL_COLUMNS == SETTINGS.max_auto_categorical_columns
    assert tools.StatsHelperAPI.MAX_CORR_COLUMNS == SETTINGS.max_corr_columns
    assert tools.StatsHelperAPI.MAX_TOP_PAIRS == SETTINGS.max_top_pairs
    assert tools.StatsHelperAPI.MAX_GROUP_ROWS == SETTINGS.max_group_rows
