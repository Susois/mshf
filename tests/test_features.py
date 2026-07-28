from mshf.line_features import line_features_from_texts
from mshf.semantic_risk import semantic_risk


def test_semantic_risk_conjunction_and_numbers():
    r = semantic_risk("A và B, ngày 2024", "A hoặc B, ngày 2025")
    assert "conjunction_and" in r["flags"]
    assert "conjunction_or" in r["flags"]
    assert "numeric_or_date_change" in r["flags"]


def test_line_features_capture_insert_modify():
    result = line_features_from_texts(
        ["dòng một", "A và B", "kết thúc"],
        ["dòng một", "dòng chèn", "A hoặc B", "kết thúc"],
    )
    assert result["ln_insert_count"] == 1
    assert result["ln_modified_count"] == 1
    assert result["ln_conjunction_count"] == 1
