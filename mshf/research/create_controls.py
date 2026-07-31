"""Tạo authentic corrupted PDFs từ original để test robustness."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from mshf.core import io_utils


def pdf_to_image(pdf_path: Path) -> np.ndarray:
    """Convert PDF trang đầu thành numpy array (dùng PyMuPDF, không cần poppler)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        # Render ở 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        # Chuyển RGBA -> RGB nếu cần
        if pix.n == 4:
            img = img[:, :, :3]
        doc.close()
        return img
    except ImportError:
        raise ImportError(
            "Can cai PyMuPDF: pip install PyMuPDF\n"
            "Hoac cai pdf2image + poppler (phuc tap hon tren Windows)"
        )


def image_to_pdf(image: np.ndarray, output_path: Path) -> None:
    """Convert numpy array về PDF."""
    img = Image.fromarray(image)
    img.convert('RGB').save(output_path, 'PDF')


def apply_jpeg_compression(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """JPEG recompression: level 1-5 (tệ hơn → tốt hơn)"""
    quality = 40 + (level - 1) * 15
    img = Image.fromarray(image)
    temp = Path("_temp_jpeg.jpg")
    img.save(temp, quality=quality)
    result = np.array(Image.open(temp))
    temp.unlink()
    return result


def apply_blur(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Gaussian blur: level 1-5 (mờ nhẹ → mờ nặng)"""
    kernel_size = 3 + (level - 1) * 2
    kernel_size = kernel_size * 2 - 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_resize(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Downsampling rồi upsample: level 1-5"""
    scale = 0.5 + (level - 1) * 0.1
    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.resize(resized, (w, h))


def apply_skew(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Rotation nhẹ: level 1-5"""
    angle = level
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_contrast(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Điều chỉnh contrast: level 1-5"""
    factor = 0.6 + (level - 1) * 0.2
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(img)
    return np.array(enhancer.enhance(factor))


def apply_noise(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Thêm Gaussian noise: level 1-5"""
    np.random.seed(seed)
    std = 5 + (level - 1) * 5
    noise = np.random.normal(0, std, image.shape)
    result = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    return result


def apply_perspective(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Perspective distortion: level 1-5"""
    h, w = image.shape[:2]
    offset = 5 + (level - 1) * 5

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [offset, 0],
        [w - offset, offset],
        [0, h - offset],
        [w - offset, h - offset]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def hash_file(path: Path) -> str:
    """SHA256 hash của file."""
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


CORRUPTION_FUNCS = {
    'jpeg': apply_jpeg_compression,
    'blur': apply_blur,
    'resize': apply_resize,
    'skew': apply_skew,
    'contrast': apply_contrast,
    'noise': apply_noise,
    'perspective': apply_perspective,
}

CORRUPTION_PARAMS = {
    'jpeg': lambda level: {'quality': 40 + (level - 1) * 15},
    'blur': lambda level: {'kernel_size': 3 + (level - 1) * 2},
    'resize': lambda level: {'scale': 0.5 + (level - 1) * 0.1},
    'skew': lambda level: {'angle': level},
    'contrast': lambda level: {'factor': 0.6 + (level - 1) * 0.2},
    'noise': lambda level: {'std': 5 + (level - 1) * 5},
    'perspective': lambda level: {'offset': 5 + (level - 1) * 5},
}


def create_corruption(
    source_pdf: Path,
    corruption_type: str,
    level: int,
    output_path: Path,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tạo corrupted PDF với provenance metadata.

    Args:
        source_pdf: đường dẫn PDF gốc
        corruption_type: 'jpeg', 'blur', 'resize', 'skew', 'contrast', 'noise', 'perspective'
        level: 1-5
        output_path: đường dẫn lưu
        seed: random seed

    Returns:
        dict với provenance metadata
    """
    image = pdf_to_image(source_pdf)
    if image is None:
        raise ValueError(f"Cannot read {source_pdf}")

    if corruption_type not in CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption: {corruption_type}")

    corrupted = CORRUPTION_FUNCS[corruption_type](image, level, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_to_pdf(corrupted, output_path)

    provenance = {
        'source_document_id': source_pdf.stem,
        'corruption': corruption_type,
        'level': level,
        'seed': seed,
        'source_path': str(source_pdf),
        'source_hash': hash_file(source_pdf),
        'output_path': str(output_path),
        'output_hash': hash_file(output_path),
        'parameters': CORRUPTION_PARAMS[corruption_type](level),
    }

    return provenance


def main():
    ap = argparse.ArgumentParser(description='Tạo authentic corrupted PDFs')
    ap.add_argument('--out-dir', type=Path, default=config.OUTPUT_DIR / 'controls')
    ap.add_argument('--max-docs', type=int, default=0, help='0=tất cả')
    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()

    doc_ids = io_utils.discover_doc_ids()
    if args.max_docs:
        doc_ids = doc_ids[:args.max_docs]

    corruptions = list(CORRUPTION_FUNCS.keys())
    levels = [1, 2, 3, 4, 5]

    import pandas as pd
    manifest = []

    for doc_id in doc_ids:
        original_path = config.PDF_ROOT / '1.original' / f'{doc_id}.pdf'
        if not original_path.exists():
            print(f'Missing: {original_path}')
            continue

        for corruption in corruptions:
            for level in levels:
                output_path = args.out_dir / corruption / str(level) / f'{doc_id}.pdf'
                try:
                    prov = create_corruption(
                        original_path, corruption, level, output_path, seed=args.seed + level
                    )
                    manifest.append(prov)
                    print(f'  ok {corruption}/{level}/{doc_id}')
                except Exception as e:
                    print(f'  FAIL {corruption}/{level}/{doc_id}: {e}')

    df = pd.DataFrame(manifest)
    manifest_path = args.out_dir / 'control_manifest.csv'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    print(f'\nManifest saved: {manifest_path}')
    print(f'Total corruptions created: {len(manifest)}')


if __name__ == '__main__':
    main()