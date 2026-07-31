from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from rapidfuzz.distance import Levenshtein
except Exception:
    Levenshtein = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not case_sensitive:
        text = text.lower()
    return text


def python_edit_distance(a: Iterable[Any], b: Iterable[Any]) -> int:
    a = list(a)
    b = list(b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def edit_distance(a: Any, b: Any) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(a, b))
    return python_edit_distance(a, b)


def compute_cer_wer(ref_text: str, hyp_text: str, *, case_sensitive: bool = False) -> dict[str, Any]:
    ref_norm = normalize_text(ref_text, case_sensitive=case_sensitive)
    hyp_norm = normalize_text(hyp_text, case_sensitive=case_sensitive)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    char_edits = edit_distance(ref_norm, hyp_norm)
    word_edits = edit_distance(ref_words, hyp_words)

    ref_chars = len(ref_norm)
    hyp_chars = len(hyp_norm)
    ref_word_count = len(ref_words)
    hyp_word_count = len(hyp_words)

    cer = char_edits / ref_chars if ref_chars else (0.0 if not hyp_chars else 1.0)
    wer = word_edits / ref_word_count if ref_word_count else (0.0 if not hyp_word_count else 1.0)

    return {
        "cer": cer,
        "wer": wer,
        "char_edits": char_edits,
        "word_edits": word_edits,
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "ref_words": ref_word_count,
        "hyp_words": hyp_word_count,
    }


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    if fitz is None:
        raise RuntimeError("Thiếu PyMuPDF. Cài bằng: pip install pymupdf")

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = out_dir / f"page_{page_index + 1:04d}.png"
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def unwrap_obj(obj: Any) -> Any:
    if isinstance(obj, (dict, list, tuple, str, int, float, type(None))):
        return obj

    for attr_name in ("json", "res", "data"):
        if hasattr(obj, attr_name):
            try:
                value = getattr(obj, attr_name)
                if callable(value):
                    value = value()
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except Exception:
                        return value
                return value
            except Exception:
                pass

    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass

    return obj


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def collect_texts_scores(obj: Any) -> tuple[list[str], list[float]]:
    obj = unwrap_obj(obj)
    texts: list[str] = []
    scores: list[float] = []

    if obj is None:
        return texts, scores

    if isinstance(obj, dict):
        if "rec_texts" in obj and isinstance(obj["rec_texts"], (list, tuple)):
            rec_texts = [str(t).strip() for t in obj.get("rec_texts", []) if str(t).strip()]
            texts.extend(rec_texts)

            rec_scores = obj.get("rec_scores", [])
            if hasattr(rec_scores, "tolist"):
                rec_scores = rec_scores.tolist()
            if isinstance(rec_scores, (list, tuple)):
                for s in rec_scores[:len(rec_texts)]:
                    try:
                        scores.append(float(s))
                    except Exception:
                        pass
            return texts, scores

        keys = ["res"] + [k for k in obj.keys() if k != "res"]
        for k in keys:
            t, s = collect_texts_scores(obj.get(k))
            texts.extend(t)
            scores.extend(s)
        return texts, scores

    if isinstance(obj, (list, tuple)):
        # PaddleOCR 2.x line format: [box, (text, score)]
        if (
            len(obj) >= 2
            and isinstance(obj[1], (list, tuple))
            and len(obj[1]) >= 2
            and isinstance(obj[1][0], str)
            and is_number(obj[1][1])
        ):
            text = obj[1][0].strip()
            if text:
                texts.append(text)
                scores.append(float(obj[1][1]))
            return texts, scores

        for item in obj:
            t, s = collect_texts_scores(item)
            texts.extend(t)
            scores.extend(s)
        return texts, scores

    return texts, scores


def build_ocr(args: argparse.Namespace) -> Any:
    from paddleocr import PaddleOCR

    try:
        # PaddleOCR 3.x
        return PaddleOCR(
            lang=args.lang,
            ocr_version=args.ocr_version,
            device=args.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=args.text_rec_score_thresh,
        )
    except TypeError:
        # PaddleOCR 2.x fallback
        use_gpu = str(args.device).lower().startswith("gpu")
        return PaddleOCR(
            lang=args.lang,
            use_angle_cls=False,
            use_gpu=use_gpu,
            show_log=False,
        )


def ocr_one_image(ocr: Any, img_path: Path) -> tuple[str, list[float]]:
    try:
        raw = ocr.predict(str(img_path))
    except AttributeError:
        raw = ocr.ocr(str(img_path), cls=False)

    texts, scores = collect_texts_scores(raw)
    return "\n".join(texts), scores


def ocr_pdf_by_rendering(pdf_path: Path, ocr: Any, args: argparse.Namespace) -> tuple[str, int, float | None]:
    with tempfile.TemporaryDirectory(prefix="vedtd_ocr_pages_") as tmp:
        page_dir = Path(tmp)
        images = render_pdf_to_images(pdf_path, page_dir, args.dpi)

        all_pages: list[str] = []
        all_scores: list[float] = []

        for img_path in images:
            page_text, page_scores = ocr_one_image(ocr, img_path)
            all_pages.append(page_text)
            all_scores.extend(page_scores)

        avg_score = mean(all_scores) if all_scores else None
        return "\n\n".join(all_pages), len(images), avg_score


def find_ground_truth(gt_root: Path, category: str, rel_pdf: Path) -> Path | None:
    direct = gt_root / category / rel_pdf.with_suffix(".txt")
    if direct.exists():
        return direct

    candidates = list((gt_root / category).rglob(rel_pdf.stem + ".txt"))
    if len(candidates) == 1:
        return candidates[0]

    return None


def safe_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR toàn bộ VEDTD bằng PaddleOCR và tính CER/WER theo từng PDF."
    )
    parser.add_argument("--root", default=".", help="Thư mục gốc VEDTD. Mặc định là thư mục hiện tại.")
    parser.add_argument("--pdf-dir", default="1.pdfs", help="Thư mục chứa PDF.")
    parser.add_argument("--gt-dir", default="2.ground_truth", help="Thư mục chứa ground truth TXT.")
    parser.add_argument("--ocr-dir", default="3.ocr_output", help="Thư mục lưu OCR TXT.")
    parser.add_argument("--report", default="ocr_eval_report.csv", help="File CSV chi tiết.")
    parser.add_argument("--summary", default="ocr_eval_summary_by_attack.csv", help="File CSV tổng hợp theo attack.")
    parser.add_argument("--lang", default="vi", help="Ngôn ngữ PaddleOCR. Tiếng Việt dùng: vi.")
    parser.add_argument("--ocr-version", default="PP-OCRv6", help="Ví dụ: PP-OCRv6 hoặc PP-OCRv5.")
    parser.add_argument("--device", default="cpu", help="cpu hoặc gpu:0.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI render PDF trước khi OCR.")
    parser.add_argument("--force", action="store_true", help="OCR lại dù đã có file TXT trong 3.ocr_output.")
    parser.add_argument("--case-sensitive", action="store_true", help="Phân biệt hoa/thường khi tính CER/WER.")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.0, help="Ngưỡng confidence OCR.")
    parser.add_argument("--max-docs-per-category", type=int, default=0, help="Test nhanh N file mỗi loại. 0 = chạy hết.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pdf_root = root / args.pdf_dir
    gt_root = root / args.gt_dir
    ocr_root = root / args.ocr_dir
    report_path = root / args.report
    summary_path = root / args.summary

    if not pdf_root.exists():
        print(f"[ERROR] Không thấy thư mục PDF: {pdf_root}", file=sys.stderr)
        return 1
    if not gt_root.exists():
        print(f"[ERROR] Không thấy thư mục ground truth: {gt_root}", file=sys.stderr)
        return 1

    print("=== VEDTD OCR CER/WER EVALUATION ===")
    print(f"ROOT       : {root}")
    print(f"PDF ROOT   : {pdf_root}")
    print(f"GT ROOT    : {gt_root}")
    print(f"OCR OUTPUT : {ocr_root}")
    print(f"DEVICE     : {args.device}")
    print(f"LANG       : {args.lang}")
    print(f"OCR VERSION: {args.ocr_version}")
    print(f"DPI        : {args.dpi}")

    ocr = build_ocr(args)
    rows: list[dict[str, Any]] = []

    for category in CATEGORIES:
        cat_pdf_dir = pdf_root / category
        if not cat_pdf_dir.exists():
            print(f"[WARN] Bỏ qua vì không thấy: {cat_pdf_dir}")
            continue

        pdf_files = sorted(cat_pdf_dir.rglob("*.pdf"))
        if args.max_docs_per_category and args.max_docs_per_category > 0:
            pdf_files = pdf_files[:args.max_docs_per_category]

        print(f"\n[{category}] {len(pdf_files)} PDF")

        for idx, pdf_path in enumerate(pdf_files, 1):
            rel_pdf = pdf_path.relative_to(cat_pdf_dir)
            gt_path = find_ground_truth(gt_root, category, rel_pdf)
            ocr_txt_path = ocr_root / category / rel_pdf.with_suffix(".txt")

            row: dict[str, Any] = {
                "category": category,
                "file": str(rel_pdf).replace("\\", "/"),
                "pdf_path": str(pdf_path),
                "ground_truth_path": str(gt_path) if gt_path else "",
                "ocr_txt_path": str(ocr_txt_path),
                "pages": "",
                "avg_rec_score": "",
                "ref_chars": "",
                "hyp_chars": "",
                "char_edits": "",
                "cer": "",
                "cer_percent": "",
                "ref_words": "",
                "hyp_words": "",
                "word_edits": "",
                "wer": "",
                "wer_percent": "",
                "status": "",
                "error": "",
            }

            try:
                if gt_path is None:
                    row["status"] = "missing_ground_truth"
                    print(f"  [{idx}/{len(pdf_files)}] MISSING GT: {rel_pdf}")
                    rows.append(row)
                    continue

                if ocr_txt_path.exists() and not args.force:
                    ocr_text = read_text(ocr_txt_path)
                    pages = ""
                    avg_score = None
                    print(f"  [{idx}/{len(pdf_files)}] cached: {rel_pdf}")
                else:
                    print(f"  [{idx}/{len(pdf_files)}] OCR: {rel_pdf}")
                    ocr_text, pages, avg_score = ocr_pdf_by_rendering(pdf_path, ocr, args)
                    write_text(ocr_txt_path, ocr_text)

                ref_text = read_text(gt_path)
                metrics = compute_cer_wer(ref_text, ocr_text, case_sensitive=args.case_sensitive)

                row.update({
                    "pages": pages,
                    "avg_rec_score": safe_float(avg_score),
                    "ref_chars": metrics["ref_chars"],
                    "hyp_chars": metrics["hyp_chars"],
                    "char_edits": metrics["char_edits"],
                    "cer": f'{metrics["cer"]:.8f}',
                    "cer_percent": f'{metrics["cer"] * 100:.4f}',
                    "ref_words": metrics["ref_words"],
                    "hyp_words": metrics["hyp_words"],
                    "word_edits": metrics["word_edits"],
                    "wer": f'{metrics["wer"]:.8f}',
                    "wer_percent": f'{metrics["wer"] * 100:.4f}',
                    "status": "ok",
                })

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                print(f"  [ERROR] {rel_pdf}: {repr(e)}", file=sys.stderr)

            rows.append(row)

    fieldnames = [
        "category", "file", "status", "error",
        "cer", "cer_percent", "wer", "wer_percent",
        "char_edits", "ref_chars", "hyp_chars",
        "word_edits", "ref_words", "hyp_words",
        "pages", "avg_rec_score",
        "pdf_path", "ground_truth_path", "ocr_txt_path",
    ]

    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[row["category"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        group = grouped.get(category, [])
        if not group:
            summary_rows.append({
                "category": category,
                "num_docs": 0,
                "macro_avg_cer_percent": "",
                "macro_avg_wer_percent": "",
                "micro_cer_percent": "",
                "micro_wer_percent": "",
                "total_char_edits": "",
                "total_ref_chars": "",
                "total_word_edits": "",
                "total_ref_words": "",
            })
            continue

        total_char_edits = sum(int(r["char_edits"]) for r in group)
        total_ref_chars = sum(int(r["ref_chars"]) for r in group)
        total_word_edits = sum(int(r["word_edits"]) for r in group)
        total_ref_words = sum(int(r["ref_words"]) for r in group)

        macro_cer = mean(float(r["cer"]) for r in group)
        macro_wer = mean(float(r["wer"]) for r in group)
        micro_cer = total_char_edits / total_ref_chars if total_ref_chars else 0.0
        micro_wer = total_word_edits / total_ref_words if total_ref_words else 0.0

        summary_rows.append({
            "category": category,
            "num_docs": len(group),
            "macro_avg_cer_percent": f"{macro_cer * 100:.4f}",
            "macro_avg_wer_percent": f"{macro_wer * 100:.4f}",
            "micro_cer_percent": f"{micro_cer * 100:.4f}",
            "micro_wer_percent": f"{micro_wer * 100:.4f}",
            "total_char_edits": total_char_edits,
            "total_ref_chars": total_ref_chars,
            "total_word_edits": total_word_edits,
            "total_ref_words": total_ref_words,
        })

    summary_fieldnames = [
        "category", "num_docs",
        "macro_avg_cer_percent", "macro_avg_wer_percent",
        "micro_cer_percent", "micro_wer_percent",
        "total_char_edits", "total_ref_chars",
        "total_word_edits", "total_ref_words",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    err_count = sum(1 for r in rows if r.get("status") == "error")
    miss_count = sum(1 for r in rows if r.get("status") == "missing_ground_truth")

    print("\n=== DONE ===")
    print(f"OK                 : {ok_count}")
    print(f"ERROR              : {err_count}")
    print(f"MISSING GT          : {miss_count}")
    print(f"Chi tiết            : {report_path}")
    print(f"Tổng hợp theo attack: {summary_path}")
    print(f"OCR text output     : {ocr_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
