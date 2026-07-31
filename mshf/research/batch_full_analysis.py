#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Full Analysis - Xử lý toàn bộ database và tạo output theo cấu trúc:
output/
  ├── 2.insert/
  ├── 3.delete/
  ├── 4.modify/
  ├── 5.layout/
  ├── summary.json
  └── summary.csv
"""
import argparse
import json
import sys
import io
from pathlib import Path
from typing import List, Dict
import csv
from datetime import datetime

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import unicodedata
import re
from rapidfuzz import fuzz

def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh"""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = 150) -> Image.Image:
    """Render một trang PDF thành ảnh"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img

def extract_all_lines(pdf_path: Path, dpi: int = 150) -> List[List[Dict]]:
    """Trích xuất text từ tất cả các trang"""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    all_pages = []
    
    for page in doc:
        blocks = page.get_text('dict')['blocks']
        lines = []
        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block.get('lines', []):
                spans = line.get('spans', [])
                text = ''.join(s['text'] for s in spans)
                if not text.strip():
                    continue
                bbox = [c * zoom for c in line['bbox']]
                lines.append({
                    "text": text.strip(),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                })
        all_pages.append(lines)
    
    doc.close()
    return all_pages

def compute_similarity_matrix(orig_lines: List[Dict], cand_lines: List[Dict]) -> np.ndarray:
    """Tính ma trận similarity"""
    n_orig = len(orig_lines)
    n_cand = len(cand_lines)
    
    similarity_matrix = np.zeros((n_orig, n_cand))
    
    for i, orig_line in enumerate(orig_lines):
        orig_text = normalize_text(orig_line['text'])
        for j, cand_line in enumerate(cand_lines):
            cand_text = normalize_text(cand_line['text'])
            similarity = fuzz.token_sort_ratio(orig_text, cand_text) / 100.0
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix

def classify_page_changes(orig_lines: List[Dict], cand_lines: List[Dict],
                         match_threshold: float = 0.8) -> Dict:
    """Phân loại thay đổi trong 1 trang"""
    if len(orig_lines) == 0 or len(cand_lines) == 0:
        return {
            'insert_count': len(cand_lines),
            'delete_count': len(orig_lines),
            'modified_count': 0,
            'matched_count': 0,
            'change_density': 1.0 if (len(orig_lines) + len(cand_lines)) > 0 else 0,
            'insert_indices': list(range(len(cand_lines))),
            'delete_indices': list(range(len(orig_lines))),
            'modified_indices': [],
            'cand_lines': cand_lines,
            'orig_lines': orig_lines
        }
    
    similarity_matrix = compute_similarity_matrix(orig_lines, cand_lines)
    
    n_orig = len(orig_lines)
    n_cand = len(cand_lines)
    
    used_orig = set()
    used_cand = set()
    matched = 0
    modified_indices = []
    
    for i in range(n_orig):
        best_j = -1
        best_sim = 0
        for j in range(n_cand):
            if similarity_matrix[i, j] > best_sim:
                best_sim = similarity_matrix[i, j]
                best_j = j
        
        if best_sim >= match_threshold and best_j not in used_cand:
            matched += 1
            used_orig.add(i)
            used_cand.add(best_j)
        elif best_sim >= 0.3:
            modified_indices.append(best_j)
            used_orig.add(i)
            used_cand.add(best_j)
    
    insert_indices = [j for j in range(n_cand) if j not in used_cand]
    delete_indices = [i for i in range(n_orig) if i not in used_orig]
    
    insert_count = len(insert_indices)
    delete_count = len(delete_indices)
    modified_count = len(modified_indices)
    
    total_lines = max(len(orig_lines), len(cand_lines))
    changes = insert_count + delete_count + modified_count
    change_density = changes / total_lines if total_lines > 0 else 0
    
    return {
        'insert_count': insert_count,
        'delete_count': delete_count,
        'modified_count': modified_count,
        'matched_count': matched,
        'change_density': change_density,
        'insert_indices': insert_indices,
        'delete_indices': delete_indices,
        'modified_indices': modified_indices,
        'cand_lines': cand_lines,
        'orig_lines': orig_lines,
        'similarity_matrix': similarity_matrix
    }

def highlight_page(image: Image.Image, page_analysis: Dict, output_path: Path):
    """Vẽ highlight trên ảnh"""
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    
    cand_lines = page_analysis['cand_lines']
    
    # RED: INSERT
    for j in page_analysis['insert_indices']:
        bbox = cand_lines[j]['bbox']
        draw.rectangle(bbox, fill=(255, 0, 0, 100), outline=(255, 0, 0, 255), width=3)
    
    # ORANGE: MODIFIED
    for j in page_analysis['modified_indices']:
        bbox = cand_lines[j]['bbox']
        draw.rectangle(bbox, fill=(255, 165, 0, 100), outline=(255, 165, 0, 255), width=2)
    
    img.save(output_path)

def analyze_and_highlight_pdf(original_pdf: Path, candidate_pdf: Path, 
                              output_dir: Path, pdf_name: str, 
                              dpi: int = 150):
    """Phân tích PDF và tạo highlighted images"""
    
    try:
        # Trích xuất text
        orig_pages = extract_all_lines(original_pdf, dpi)
        cand_pages = extract_all_lines(candidate_pdf, dpi)
        
        max_pages = max(len(orig_pages), len(cand_pages))
        
        total_insert = 0
        total_delete = 0
        total_modified = 0
        total_matched = 0
        
        # Xử lý từng trang
        for page_idx in range(max_pages):
            orig_lines = orig_pages[page_idx] if page_idx < len(orig_pages) else []
            cand_lines = cand_pages[page_idx] if page_idx < len(cand_pages) else []
            
            # Phân loại
            page_analysis = classify_page_changes(orig_lines, cand_lines)
            
            total_insert += page_analysis['insert_count']
            total_delete += page_analysis['delete_count']
            total_modified += page_analysis['modified_count']
            total_matched += page_analysis['matched_count']
            
            # Render và highlight
            cand_img = render_pdf_page(candidate_pdf, page_idx, dpi)
            output_img = output_dir / f"page_{page_idx:03d}_highlighted.png"
            highlight_page(cand_img, page_analysis, output_img)
        
        return {
            'success': True,
            'insert': total_insert,
            'delete': total_delete,
            'modified': total_modified,
            'matched': total_matched,
            'pages': max_pages,
            'output_dir': str(output_dir)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Batch full analysis and highlight")
    parser.add_argument("--database-dir", default="Tuan1_2/VEDTD/1.pdfs", help="Database directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--dpi", type=int, default=150, help="DPI")
    
    args = parser.parse_args()
    
    database_dir = Path(args.database_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Attack types
    attack_types = {
        '1.original': 'authentic',
        '2.insert': '2.insert',
        '3.delete': '3.delete',
        '4.modify': '4.modify',
        '5.layout': '5.layout'
    }
    
    # Tìm original PDFs
    original_dir = database_dir / '1.original'
    if not original_dir.exists():
        print(f"[ERROR] Không tìm thấy {original_dir}")
        return 1
    
    original_pdfs = list(original_dir.glob("*.pdf"))
    print(f"Tìm thấy {len(original_pdfs)} PDF gốc")
    
    results = []
    total_count = 0
    processed_count = 0
    
    print(f"\n{'='*70}")
    print("BẮT ĐẦU BATCH FULL ANALYSIS")
    print(f"{'='*70}\n")
    
    for orig_pdf in original_pdfs:
        pdf_name = orig_pdf.stem
        print(f"\n[{processed_count+1}/{len(original_pdfs)}] Processing: {pdf_name}")
        
        for attack_label, attack_dir_name in attack_types.items():
            if attack_label == '1.original':
                continue
            
            attack_dir = database_dir / attack_label
            if not attack_dir.exists():
                continue
            
            candidate_pdf = attack_dir / orig_pdf.name
            if not candidate_pdf.exists():
                continue
            
            total_count += 1
            
            # Tạo output folder
            pdf_output_dir = output_dir / attack_dir_name / pdf_name
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Phân tích
            result = analyze_and_highlight_pdf(
                orig_pdf, candidate_pdf, pdf_output_dir, pdf_name, args.dpi
            )
            
            result['pdf_name'] = pdf_name
            result['attack_type'] = attack_dir_name
            
            if result['success']:
                print(f"  ✓ {attack_label}: {result['insert']}I {result['delete']}D {result['modified']}M")
                processed_count += 1
            else:
                print(f"  ✗ {attack_label}: {result['error']}")
            
            results.append(result)
    
    # Tạo summary JSON
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'database_dir': str(database_dir),
        'output_dir': str(output_dir),
        'total_processed': processed_count,
        'total_count': total_count,
        'results': [r for r in results if r.get('success')]
    }
    
    summary_json_path = output_dir / 'summary.json'
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    
    # Tạo summary CSV
    summary_csv_path = output_dir / 'summary.csv'
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['pdf_name', 'attack_type', 'insert', 'delete', 'modified', 'matched', 'pages'])
        writer.writeheader()
        for r in results:
            if r.get('success'):
                writer.writerow({
                    'pdf_name': r['pdf_name'],
                    'attack_type': r['attack_type'],
                    'insert': r['insert'],
                    'delete': r['delete'],
                    'modified': r['modified'],
                    'matched': r['matched'],
                    'pages': r['pages']
                })
    
    print(f"\n{'='*70}")
    print("HOÀN THÀNH")
    print(f"{'='*70}")
    print(f"Processed: {processed_count}/{total_count}")
    print(f"Output: {output_dir}/")
    print(f"Summary: {summary_json_path}")
    print(f"CSV: {summary_csv_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())