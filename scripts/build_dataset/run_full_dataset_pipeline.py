from pathlib import Path
import shutil
import csv
import traceback
import time
import gc

import pythoncom
import win32com.client as win32


# ============================================================
# CẤU HÌNH THƯ MỤC
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCX_DIR = BASE_DIR / "6.docx"

# PDF output folders
PDF_ORIGINAL_DIR = BASE_DIR / "1.original"
PDF_INSERT_DIR = BASE_DIR / "2.insert"
PDF_DELETE_DIR = BASE_DIR / "3.delete"
PDF_MODIFY_DIR = BASE_DIR / "4.modify"
PDF_LAYOUT_DIR = BASE_DIR / "5.layout"

# Ground truth folders
GT_DIR = BASE_DIR / "ground_truth"
GT_ORIGINAL_DIR = GT_DIR / "1.original"
GT_INSERT_DIR = GT_DIR / "2.insert"
GT_DELETE_DIR = GT_DIR / "3.delete"
GT_MODIFY_DIR = GT_DIR / "4.modify"
GT_LAYOUT_DIR = GT_DIR / "5.layout"

TEMP_DIR = BASE_DIR / "_temp_full_pipeline_docx"

REPORT_CSV = BASE_DIR / "full_pipeline_report.csv"
MANIFEST_CSV = BASE_DIR / "manifest.csv"

OVERWRITE_EXISTING = True
MAX_RETRIES_PER_FILE = 3


# ============================================================
# INSERT ATTACK CONFIG
# ============================================================

INSERT_TEXT = """
– Công ty X có thể xác lập quyền sử dụng đất để thực hiện dự án này bằng những cách sau:
+ Thông qua việc Nhà nước giao đất để thực hiện các dự án xây dựng nhà ở để bán theo quy định tại điểm i khoản 1 Điều 28 Luật Đất đai 2024 và khoản 3 Điều 119 Luật Đất đai 2024.
+ Thông qua việc Nhà nước cho thuê đất theo quy định tại điểm k khoản 1 Điều 28 Luật Đất đai 2024 và Điều 120 Luật Đất đai 2024.
+ Thông qua việc nhận chuyển nhượng vốn đầu tư là giá trị quyền sử dụng đất theo quy định tại điểm d khoản 1 Điều 28 Luật Đất đai 2024;
+ Thông qua việc thuê lại đất trong khu công nghiệp, cụm công nghiệp, khu chế xuất, khu công nghệ cao, khu kinh tế theo quy định tại khoản 2 Điều 43 Luật Đất đai 2024.
"""

# 0.7 = chèn gần phía dưới của trang 1
INSERT_RATIO_ON_PAGE_1 = 0.7

INSERT_FONT_NAME = "Times New Roman"
INSERT_FONT_SIZE = 12


# ============================================================
# DELETE ATTACK CONFIG
# ============================================================

# Xóa 3 dòng đầu
DELETE_LINES = 3

# Hiện tại đang xóa 3 dòng đầu của file, tức là trang 1.
# Nếu cần quay lại xóa 3 dòng đầu trang 2, đổi DELETE_PAGE_NUMBER = 2
DELETE_PAGE_NUMBER = 2


# ============================================================
# MODIFY ATTACK CONFIG
# ============================================================

FIND_TEXT = "và"
REPLACE_TEXT = "hoặc"

# True = chỉ thay từ độc lập "và"
MATCH_WHOLE_WORD = True

# True = chỉ thay đúng chữ thường "và", không thay "Và"
MATCH_CASE = True


# ============================================================
# LAYOUT ATTACK CONFIG
# ============================================================

TARGET_FONT_NAME = "Arial"
TARGET_FONT_SIZE = 10


# ============================================================
# WORD CONSTANTS
# ============================================================

WD_ACTIVE_END_PAGE_NUMBER = 3
WD_GO_TO_PAGE = 1
WD_GO_TO_ABSOLUTE = 1
WD_LINE = 5
WD_EXTEND = 1
WD_DO_NOT_SAVE_CHANGES = 0
WD_EXPORT_FORMAT_PDF = 17
WD_REPLACE_ALL = 2
WD_FIND_STOP = 0
WD_LINE_SPACE_1_5 = 1
WD_STATISTIC_PAGES = 2


# ============================================================
# TIỆN ÍCH CHUNG
# ============================================================

def ensure_dirs():
    folders = [
        PDF_ORIGINAL_DIR,
        PDF_INSERT_DIR,
        PDF_DELETE_DIR,
        PDF_MODIFY_DIR,
        PDF_LAYOUT_DIR,
        GT_ORIGINAL_DIR,
        GT_INSERT_DIR,
        GT_DELETE_DIR,
        GT_MODIFY_DIR,
        GT_LAYOUT_DIR,
        TEMP_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\x07", "")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def save_doc_text_to_txt(doc, txt_path: Path):
    text = doc.Content.Text
    text = normalize_text(text)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")


def export_doc_to_pdf(doc, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if pdf_path.exists() and OVERWRITE_EXISTING:
        for _ in range(10):
            try:
                pdf_path.unlink()
                break
            except PermissionError:
                time.sleep(0.5)

    if pdf_path.exists() and not OVERWRITE_EXISTING:
        return

    doc.ExportAsFixedFormat(
        OutputFileName=str(pdf_path),
        ExportFormat=WD_EXPORT_FORMAT_PDF,
        OpenAfterExport=False
    )

    time.sleep(0.5)


def safe_delete_file(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def close_doc_without_save(doc):
    if doc is None:
        return

    for _ in range(3):
        try:
            doc.Close(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
            return
        except Exception:
            time.sleep(1)


def open_temp_doc(word, src_docx_path: Path, variant_name: str):
    temp_docx_path = TEMP_DIR / f"{src_docx_path.stem}_{variant_name}.docx"

    safe_delete_file(temp_docx_path)
    shutil.copy2(src_docx_path, temp_docx_path)

    doc = word.Documents.Open(
        str(temp_docx_path),
        ReadOnly=False,
        AddToRecentFiles=False,
        Visible=False
    )

    time.sleep(0.3)

    return doc, temp_docx_path


def start_word_app():
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    try:
        word.ScreenUpdating = False
    except Exception:
        pass

    try:
        word.Options.SaveInterval = 0
    except Exception:
        pass

    time.sleep(1)

    return word


def stop_word_app(word):
    if word is None:
        return

    try:
        word.Documents.Close(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
    except Exception:
        pass

    time.sleep(0.5)

    try:
        word.Quit()
    except Exception:
        pass

    time.sleep(1)


def is_call_rejected_error(error):
    error_text = str(error)
    return (
        "Call was rejected by callee" in error_text
        or "-2147418111" in error_text
        or "RPC_E_CALL_REJECTED" in error_text
    )


def count_text_in_doc(doc, target: str):
    try:
        return doc.Content.Text.count(target)
    except Exception:
        return ""


def write_csv(csv_path: Path, fieldnames, rows):
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# ORIGINAL
# ============================================================

def create_original(word, src_docx_path: Path):
    doc = None
    temp_path = None

    txt_path = GT_ORIGINAL_DIR / f"{src_docx_path.stem}.txt"
    pdf_path = PDF_ORIGINAL_DIR / f"{src_docx_path.stem}.pdf"

    try:
        doc, temp_path = open_temp_doc(word, src_docx_path, "original")

        save_doc_text_to_txt(doc, txt_path)
        export_doc_to_pdf(doc, pdf_path)

        return {
            "status": "success",
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "extra": ""
        }

    finally:
        close_doc_without_save(doc)
        if temp_path:
            safe_delete_file(temp_path)


# ============================================================
# INSERT ATTACK
# ============================================================

def get_page_number(paragraph):
    return paragraph.Range.Information(WD_ACTIVE_END_PAGE_NUMBER)


def get_paragraphs_on_page(doc, page_number: int):
    doc.Repaginate()

    paragraphs = []

    for i in range(1, doc.Paragraphs.Count + 1):
        para = doc.Paragraphs(i)

        text = para.Range.Text.strip()
        if not text:
            continue

        try:
            current_page = get_page_number(para)
        except Exception:
            continue

        if current_page == page_number:
            paragraphs.append(para)

        if current_page > page_number and len(paragraphs) > 0:
            break

    return paragraphs


def choose_insert_paragraph_on_page_1(doc):
    paragraphs = get_paragraphs_on_page(doc, 1)

    if not paragraphs:
        return doc.Paragraphs(1)

    index = int(len(paragraphs) * INSERT_RATIO_ON_PAGE_1)

    if index < 0:
        index = 0

    if index >= len(paragraphs):
        index = len(paragraphs) - 1

    return paragraphs[index]


def apply_insert_attack(doc):
    target_para = choose_insert_paragraph_on_page_1(doc)

    insert_block = INSERT_TEXT.strip() + "\r\r"
    start_pos = target_para.Range.Start

    insert_range = doc.Range(start_pos, start_pos)
    insert_range.InsertBefore(insert_block)

    inserted_range = doc.Range(start_pos, start_pos + len(insert_block))
    inserted_range.Font.Name = INSERT_FONT_NAME
    inserted_range.Font.Size = INSERT_FONT_SIZE

    doc.Repaginate()


def create_insert(word, src_docx_path: Path):
    doc = None
    temp_path = None

    txt_path = GT_INSERT_DIR / f"{src_docx_path.stem}.txt"
    pdf_path = PDF_INSERT_DIR / f"{src_docx_path.stem}.pdf"

    try:
        doc, temp_path = open_temp_doc(word, src_docx_path, "insert")

        apply_insert_attack(doc)

        save_doc_text_to_txt(doc, txt_path)
        export_doc_to_pdf(doc, pdf_path)

        return {
            "status": "success",
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "extra": ""
        }

    finally:
        close_doc_without_save(doc)
        if temp_path:
            safe_delete_file(temp_path)


# ============================================================
# DELETE ATTACK
# ============================================================

def get_page_count(doc):
    doc.Repaginate()
    return doc.ComputeStatistics(WD_STATISTIC_PAGES)


def apply_delete_attack(word, doc):
    doc.Repaginate()

    total_pages = get_page_count(doc)

    if total_pages < DELETE_PAGE_NUMBER:
        raise RuntimeError(
            f"File chỉ có {total_pages} trang, không có trang {DELETE_PAGE_NUMBER}"
        )

    doc.Activate()
    time.sleep(0.3)

    selection = word.Selection

    selection.GoTo(
        What=WD_GO_TO_PAGE,
        Which=WD_GO_TO_ABSOLUTE,
        Count=DELETE_PAGE_NUMBER
    )

    start_pos = selection.Range.Start
    selection.SetRange(start_pos, start_pos)

    selection.MoveDown(
        Unit=WD_LINE,
        Count=DELETE_LINES,
        Extend=WD_EXTEND
    )

    selected_text = selection.Text

    if not selected_text.strip():
        raise RuntimeError(
            f"Không chọn được {DELETE_LINES} dòng đầu của trang {DELETE_PAGE_NUMBER}"
        )

    selection.Delete()
    doc.Repaginate()

    return selected_text


def create_delete(word, src_docx_path: Path):
    doc = None
    temp_path = None

    txt_path = GT_DELETE_DIR / f"{src_docx_path.stem}.txt"
    pdf_path = PDF_DELETE_DIR / f"{src_docx_path.stem}.pdf"

    try:
        doc, temp_path = open_temp_doc(word, src_docx_path, "delete")

        deleted_text = apply_delete_attack(word, doc)

        save_doc_text_to_txt(doc, txt_path)
        export_doc_to_pdf(doc, pdf_path)

        return {
            "status": "success",
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "extra": normalize_text(deleted_text).replace("\n", " | ")
        }

    finally:
        close_doc_without_save(doc)
        if temp_path:
            safe_delete_file(temp_path)


# ============================================================
# MODIFY ATTACK
# ============================================================

def apply_modify_attack(doc):
    find = doc.Content.Find

    find.ClearFormatting()
    find.Replacement.ClearFormatting()

    find.Text = FIND_TEXT
    find.Replacement.Text = REPLACE_TEXT

    find.Forward = True
    find.Wrap = WD_FIND_STOP
    find.Format = False
    find.MatchCase = MATCH_CASE
    find.MatchWholeWord = MATCH_WHOLE_WORD
    find.MatchWildcards = False
    find.MatchSoundsLike = False
    find.MatchAllWordForms = False

    find.Execute(
        FindText=FIND_TEXT,
        MatchCase=MATCH_CASE,
        MatchWholeWord=MATCH_WHOLE_WORD,
        MatchWildcards=False,
        MatchSoundsLike=False,
        MatchAllWordForms=False,
        Forward=True,
        Wrap=WD_FIND_STOP,
        Format=False,
        ReplaceWith=REPLACE_TEXT,
        Replace=WD_REPLACE_ALL
    )

    doc.Repaginate()


def create_modify(word, src_docx_path: Path):
    doc = None
    temp_path = None

    txt_path = GT_MODIFY_DIR / f"{src_docx_path.stem}.txt"
    pdf_path = PDF_MODIFY_DIR / f"{src_docx_path.stem}.pdf"

    try:
        doc, temp_path = open_temp_doc(word, src_docx_path, "modify")

        before_count = count_text_in_doc(doc, FIND_TEXT)

        apply_modify_attack(doc)

        after_count = count_text_in_doc(doc, FIND_TEXT)

        save_doc_text_to_txt(doc, txt_path)
        export_doc_to_pdf(doc, pdf_path)

        replaced_count = ""
        if isinstance(before_count, int) and isinstance(after_count, int):
            replaced_count = before_count - after_count

        return {
            "status": "success",
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "extra": f"before={before_count}; after={after_count}; replaced_estimate={replaced_count}"
        }

    finally:
        close_doc_without_save(doc)
        if temp_path:
            safe_delete_file(temp_path)


# ============================================================
# LAYOUT ATTACK
# ============================================================

def format_range_text(rng):
    try:
        rng.Font.Name = TARGET_FONT_NAME
        rng.Font.NameAscii = TARGET_FONT_NAME
        rng.Font.NameBi = TARGET_FONT_NAME
        rng.Font.NameFarEast = TARGET_FONT_NAME
        rng.Font.NameOther = TARGET_FONT_NAME
        rng.Font.Size = TARGET_FONT_SIZE
        rng.ParagraphFormat.LineSpacingRule = WD_LINE_SPACE_1_5
    except Exception:
        pass


def format_shapes_text(doc):
    try:
        for shape in doc.Shapes:
            try:
                if shape.TextFrame.HasText:
                    format_range_text(shape.TextFrame.TextRange)
            except Exception:
                pass
    except Exception:
        pass


def apply_layout_attack(doc):
    format_range_text(doc.Content)

    try:
        story_ranges = doc.StoryRanges

        for i in range(1, story_ranges.Count + 1):
            rng = story_ranges.Item(i)

            while rng is not None:
                format_range_text(rng)

                try:
                    rng = rng.NextStoryRange
                except Exception:
                    rng = None

    except Exception:
        pass

    try:
        for table in doc.Tables:
            format_range_text(table.Range)
    except Exception:
        pass

    format_shapes_text(doc)

    doc.Repaginate()


def create_layout(word, src_docx_path: Path):
    doc = None
    temp_path = None

    txt_path = GT_LAYOUT_DIR / f"{src_docx_path.stem}.txt"
    pdf_path = PDF_LAYOUT_DIR / f"{src_docx_path.stem}.pdf"

    try:
        doc, temp_path = open_temp_doc(word, src_docx_path, "layout")

        apply_layout_attack(doc)

        save_doc_text_to_txt(doc, txt_path)
        export_doc_to_pdf(doc, pdf_path)

        return {
            "status": "success",
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "extra": f"font={TARGET_FONT_NAME}; size={TARGET_FONT_SIZE}; line_spacing=1.5"
        }

    finally:
        close_doc_without_save(doc)
        if temp_path:
            safe_delete_file(temp_path)


# ============================================================
# XỬ LÝ 1 FILE
# ============================================================

def process_one_file_once(docx_path: Path):
    word = None

    try:
        word = start_word_app()

        doc_id = docx_path.stem

        result = {
            "file": docx_path.name,
            "original_status": "",
            "insert_status": "",
            "delete_status": "",
            "modify_status": "",
            "layout_status": "",
            "original_txt": "",
            "insert_txt": "",
            "delete_txt": "",
            "modify_txt": "",
            "layout_txt": "",
            "original_pdf": "",
            "insert_pdf": "",
            "delete_pdf": "",
            "modify_pdf": "",
            "layout_pdf": "",
            "delete_extra": "",
            "modify_extra": "",
            "layout_extra": "",
            "error": "",
        }

        print("    - Original")
        original = create_original(word, docx_path)
        result["original_status"] = original["status"]
        result["original_txt"] = original["txt_path"]
        result["original_pdf"] = original["pdf_path"]

        print("    - Insert")
        insert = create_insert(word, docx_path)
        result["insert_status"] = insert["status"]
        result["insert_txt"] = insert["txt_path"]
        result["insert_pdf"] = insert["pdf_path"]

        print("    - Delete")
        delete = create_delete(word, docx_path)
        result["delete_status"] = delete["status"]
        result["delete_txt"] = delete["txt_path"]
        result["delete_pdf"] = delete["pdf_path"]
        result["delete_extra"] = delete["extra"]

        print("    - Modify")
        modify = create_modify(word, docx_path)
        result["modify_status"] = modify["status"]
        result["modify_txt"] = modify["txt_path"]
        result["modify_pdf"] = modify["pdf_path"]
        result["modify_extra"] = modify["extra"]

        print("    - Layout")
        layout = create_layout(word, docx_path)
        result["layout_status"] = layout["status"]
        result["layout_txt"] = layout["txt_path"]
        result["layout_pdf"] = layout["pdf_path"]
        result["layout_extra"] = layout["extra"]

        manifest_rows = [
            {
                "doc_id": doc_id,
                "variant_id": f"{doc_id}_original",
                "attack_type": "original",
                "label": 0,
                "source_docx": str(docx_path),
                "pdf_path": result["original_pdf"],
                "gt_path": result["original_txt"],
                "status": result["original_status"],
            },
            {
                "doc_id": doc_id,
                "variant_id": f"{doc_id}_insert",
                "attack_type": "insert",
                "label": 1,
                "source_docx": str(docx_path),
                "pdf_path": result["insert_pdf"],
                "gt_path": result["insert_txt"],
                "status": result["insert_status"],
            },
            {
                "doc_id": doc_id,
                "variant_id": f"{doc_id}_delete",
                "attack_type": "delete",
                "label": 1,
                "source_docx": str(docx_path),
                "pdf_path": result["delete_pdf"],
                "gt_path": result["delete_txt"],
                "status": result["delete_status"],
            },
            {
                "doc_id": doc_id,
                "variant_id": f"{doc_id}_modify",
                "attack_type": "modify",
                "label": 1,
                "source_docx": str(docx_path),
                "pdf_path": result["modify_pdf"],
                "gt_path": result["modify_txt"],
                "status": result["modify_status"],
            },
            {
                "doc_id": doc_id,
                "variant_id": f"{doc_id}_layout",
                "attack_type": "layout",
                "label": 1,
                "source_docx": str(docx_path),
                "pdf_path": result["layout_pdf"],
                "gt_path": result["layout_txt"],
                "status": result["layout_status"],
            },
        ]

        return result, manifest_rows

    finally:
        stop_word_app(word)
        gc.collect()


def process_one_file_with_retry(docx_path: Path, max_retries: int = MAX_RETRIES_PER_FILE):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return process_one_file_once(docx_path)

        except Exception as e:
            last_error = e

            wait_seconds = attempt * 4

            if is_call_rejected_error(e):
                print(
                    f"  -> Word đang bận: retry {attempt}/{max_retries} "
                    f"sau {wait_seconds}s..."
                )
            else:
                print(
                    f"  -> Lỗi khi xử lý file: retry {attempt}/{max_retries} "
                    f"sau {wait_seconds}s..."
                )
                print(f"     Chi tiết: {str(e)[:300]}")

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Không xử lý được file sau {max_retries} lần retry: {docx_path.name}\n"
        f"Lỗi cuối: {last_error}\n"
        f"{traceback.format_exc()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not DOCX_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder: {DOCX_DIR}")

    ensure_dirs()

    docx_files = sorted([
        p for p in DOCX_DIR.glob("*.docx")
        if not p.name.startswith("~$")
    ])

    print(f"Tìm thấy {len(docx_files)} file DOCX trong: {DOCX_DIR}")
    print("Bắt đầu tạo dataset...")

    pythoncom.CoInitialize()

    report_rows = []
    manifest_rows_all = []

    report_fieldnames = [
        "file",
        "original_status",
        "insert_status",
        "delete_status",
        "modify_status",
        "layout_status",
        "original_txt",
        "insert_txt",
        "delete_txt",
        "modify_txt",
        "layout_txt",
        "original_pdf",
        "insert_pdf",
        "delete_pdf",
        "modify_pdf",
        "layout_pdf",
        "delete_extra",
        "modify_extra",
        "layout_extra",
        "error",
    ]

    manifest_fieldnames = [
        "doc_id",
        "variant_id",
        "attack_type",
        "label",
        "source_docx",
        "pdf_path",
        "gt_path",
        "status",
    ]

    try:
        for idx, docx_path in enumerate(docx_files, start=1):
            print(f"\n[{idx}/{len(docx_files)}] Đang xử lý: {docx_path.name}")

            try:
                result, manifest_rows = process_one_file_with_retry(
                    docx_path,
                    max_retries=MAX_RETRIES_PER_FILE
                )

                report_rows.append(result)
                manifest_rows_all.extend(manifest_rows)

                statuses = [
                    result["original_status"],
                    result["insert_status"],
                    result["delete_status"],
                    result["modify_status"],
                    result["layout_status"],
                ]

                if all(s == "success" for s in statuses):
                    print("  -> OK toàn bộ")
                else:
                    print("  -> Có lỗi một số nhánh:", statuses)

            except Exception as e:
                error_text = str(e) + "\n" + traceback.format_exc()
                print("  -> LỖI FILE:", str(e)[:500])

                failed_result = {
                    "file": docx_path.name,
                    "original_status": "failed",
                    "insert_status": "failed",
                    "delete_status": "failed",
                    "modify_status": "failed",
                    "layout_status": "failed",
                    "original_txt": "",
                    "insert_txt": "",
                    "delete_txt": "",
                    "modify_txt": "",
                    "layout_txt": "",
                    "original_pdf": "",
                    "insert_pdf": "",
                    "delete_pdf": "",
                    "modify_pdf": "",
                    "layout_pdf": "",
                    "delete_extra": "",
                    "modify_extra": "",
                    "layout_extra": "",
                    "error": error_text,
                }

                report_rows.append(failed_result)

                time.sleep(3)

            # Ghi report sau mỗi file để nếu script bị dừng giữa chừng vẫn còn log
            write_csv(REPORT_CSV, report_fieldnames, report_rows)
            write_csv(MANIFEST_CSV, manifest_fieldnames, manifest_rows_all)

    finally:
        pythoncom.CoUninitialize()

    print("\nHoàn tất.")
    print(f"Report: {REPORT_CSV}")
    print(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()