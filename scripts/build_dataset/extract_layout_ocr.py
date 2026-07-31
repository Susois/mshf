# extract_layout_ocr.py
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


# ---------------------------------------------------------------------------
# Render PDF -> ảnh (giữ nguyên logic từ file CER/WER cũ)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Trích text + bbox (khác bản cũ: bản cũ chỉ lấy text+score, không lấy bbox)
# ---------------------------------------------------------------------------
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


def to_rect(box: Any) -> list[float] | None:
    """Chuẩn hóa box về [x0, y0, x1, y1]. Nhận cả dạng rect có sẵn hoặc quad 4 điểm."""
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    try:
        if len(box) == 4 and all(is_number(v) for v in box):
            x0, y0, x1, y1 = box
            return [float(x0), float(y0), float(x1), float(y1)]
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def collect_lines(obj: Any) -> list[dict[str, Any]]:
    """Tương tự collect_texts_scores ở bản cũ, nhưng giữ lại bbox của mỗi dòng."""
    obj = unwrap_obj(obj)
    lines: list[dict[str, Any]] = []

    if obj is None:
        return lines

    if isinstance(obj, dict):
        if "rec_texts" in obj and isinstance(obj["rec_texts"], (list, tuple)):
            texts = [str(t).strip() for t in obj.get("rec_texts", [])]

            scores = obj.get("rec_scores", [])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            # PaddleOCR 3.x: rec_boxes (rect) ưu tiên hơn, fallback rec_polys/dt_polys (quad)
            boxes = obj.get("rec_boxes", None)
            if boxes is None:
                boxes = obj.get("rec_polys", None)
            if boxes is None:
                boxes = obj.get("dt_polys", None)
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            if boxes is None:
                boxes = [None] * len(texts)

            for i, text in enumerate(texts):
                if not text:
                    continue
                score = None
                if i < len(scores):
                    try:
                        score = float(scores[i])
                    except Exception:
                        pass
                bbox = to_rect(boxes[i]) if i < len(boxes) else None
                lines.append({"text": text, "bbox": bbox, "score": score})
            return lines

        keys = ["res"] + [k for k in obj.keys() if k != "res"]
        for k in keys:
            lines.extend(collect_lines(obj.get(k)))
        return lines

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
                bbox = to_rect(obj[0])
                lines.append({"text": text, "bbox": bbox, "score": float(obj[1][1])})
            return lines

        for item in obj:
            lines.extend(collect_lines(item))
        return lines

    return lines


def ocr_one_image_layout(ocr: Any, img_path: Path) -> list[dict[str, Any]]:
    try:
        raw = ocr.predict(str(img_path))
    except AttributeError:
        raw = ocr.ocr(str(img_path), cls=False)
    return collect_lines(raw)


def get_image_size(img_path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(img_path) as im:
            return im.width, im.height
    # fallback nếu thiếu Pillow: đọc lại bằng fitz pixmap đã lưu kích thước trong tên file không đáng tin,
    # nên cứ raise để báo cài Pillow
    raise RuntimeError("Thiếu Pillow để đọc kích thước ảnh. Cài bằng: pip install pillow")


def ocr_pdf_layout(pdf_path: Path, ocr: Any, args: argparse.Namespace) -> dict[str, Any]:
    """OCR toàn bộ PDF, chỉ giữ lại text + bbox + score cho từng dòng, theo từng trang."""
    with tempfile.TemporaryDirectory(prefix="vedtd_layout_pages_") as tmp:
        page_dir = Path(tmp)
        images = render_pdf_to_images(pdf_path, page_dir, args.dpi)

        pages_out: list[dict[str, Any]] = []
        all_scores: list[float] = []

        for page_index, img_path in enumerate(images):
            width, height = get_image_size(img_path)
            lines = ocr_one_image_layout(ocr, img_path)
            for line in lines:
                if line.get("score") is not None:
                    all_scores.append(line["score"])
            pages_out.append({
                "page": page_index,
                "page_width": width,
                "page_height": height,
                "dpi": args.dpi,
                "lines": lines,
            })

        avg_score = mean(all_scores) if all_scores else None
        return {
            "pdf_path": str(pdf_path),
            "num_pages": len(images),
            "avg_rec_score": avg_score,
            "pages": pages_out,
        }


def find_existing_json(out_path: Path) -> bool:
    return out_path.exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR toàn bộ VEDTD bằng PaddleOCR, chỉ để trích layout (text + bbox), không lưu .txt và không tính CER/WER."
    )
    parser.add_argument("--root", default=".", help="Thư mục gốc VEDTD.")
    parser.add_argument("--pdf-dir", default="1.pdfs", help="Thư mục chứa PDF.")
    parser.add_argument("--layout-dir", default="4.layout_ocr", help="Thư mục lưu JSON layout.")
    parser.add_argument("--report", default="layout_extraction_report.csv", help="File CSV trạng thái trích layout.")
    parser.add_argument("--lang", default="vi", help="Ngôn ngữ PaddleOCR. Tiếng Việt dùng: vi.")
    parser.add_argument("--ocr-version", default="PP-OCRv6", help="Ví dụ: PP-OCRv6 hoặc PP-OCRv5.")
    parser.add_argument("--device", default="cpu", help="cpu hoặc gpu:0.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI render PDF trước khi OCR.")
    parser.add_argument("--force", action="store_true", help="Trích lại dù đã có JSON.")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.0, help="Ngưỡng confidence OCR.")
    parser.add_argument("--max-docs-per-category", type=int, default=0, help="Test nhanh N file mỗi loại. 0 = chạy hết.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pdf_root = root / args.pdf_dir
    layout_root = root / args.layout_dir
    report_path = root / args.report

    if not pdf_root.exists():
        print(f"[ERROR] Không thấy thư mục PDF: {pdf_root}", file=sys.stderr)
        return 1

    print("=== VEDTD LAYOUT EXTRACTION (PaddleOCR) ===")
    print(f"ROOT       : {root}")
    print(f"PDF ROOT   : {pdf_root}")
    print(f"LAYOUT OUT : {layout_root}")
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
            out_json_path = layout_root / category / rel_pdf.with_suffix(".json")

            row: dict[str, Any] = {
                "category": category,
                "file": str(rel_pdf).replace("\\", "/"),
                "pdf_path": str(pdf_path),
                "layout_json_path": str(out_json_path),
                "num_pages": "",
                "num_lines": "",
                "avg_rec_score": "",
                "status": "",
                "error": "",
            }

            try:
                if find_existing_json(out_json_path) and not args.force:
                    print(f"  [{idx}/{len(pdf_files)}] cached: {rel_pdf}")
                    with out_json_path.open("r", encoding="utf-8") as f:
                        cached = json.load(f)
                    row["num_pages"] = cached.get("num_pages", "")
                    row["num_lines"] = sum(len(p.get("lines", [])) for p in cached.get("pages", []))
                    row["avg_rec_score"] = cached.get("avg_rec_score", "")
                    row["status"] = "cached"
                    rows.append(row)
                    continue

                print(f"  [{idx}/{len(pdf_files)}] OCR layout: {rel_pdf}")
                result = ocr_pdf_layout(pdf_path, ocr, args)

                out_json_path.parent.mkdir(parents=True, exist_ok=True)
                with out_json_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                row["num_pages"] = result["num_pages"]
                row["num_lines"] = sum(len(p["lines"]) for p in result["pages"])
                row["avg_rec_score"] = (
                    f'{result["avg_rec_score"]:.6f}' if result["avg_rec_score"] is not None else ""
                )
                row["status"] = "ok"

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                print(f"  [ERROR] {rel_pdf}: {repr(e)}", file=sys.stderr)

            rows.append(row)

    fieldnames = [
        "category", "file", "status", "error",
        "num_pages", "num_lines", "avg_rec_score",
        "pdf_path", "layout_json_path",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r.get("status") in ("ok", "cached"))
    err_count = sum(1 for r in rows if r.get("status") == "error")

    print("\n=== DONE ===")
    print(f"OK/CACHED : {ok_count}")
    print(f"ERROR     : {err_count}")
    print(f"Báo cáo   : {report_path}")
    print(f"Layout JSON: {layout_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())