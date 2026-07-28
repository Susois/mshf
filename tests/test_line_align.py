from mshf.line_align import line_cer, match_lines
from mshf.io_utils import as_line_dicts


def test_line_cer_identical_and_changed():
    assert line_cer("Xin chào", "xin chào") == 0
    assert line_cer("và", "hoặc") > 0


def test_alignment_insert_and_modify():
    ref = as_line_dicts(["dòng một", "A và B", "kết thúc"])
    cand = as_line_dicts(["dòng một", "dòng chèn", "A hoặc B", "kết thúc"])
    pairs = match_lines(ref, cand)
    assert sum(p["type"] == "inserted" for p in pairs) == 1
    changed = [p for p in pairs if p["type"] == "match" and p["cer"] > 0]
    assert len(changed) == 1
