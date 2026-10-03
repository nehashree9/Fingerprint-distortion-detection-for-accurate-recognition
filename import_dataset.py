"""
Dataset Importer for Family Fingerprint Dataset into MySQL with SourceAFIS.
Recursively scans DATA/FAMILY FINGERPRINT DATASET, validates images, extracts metadata,
generates SourceAFIS templates, and stores records in MySQL.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# Ensure root directory is on Python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from database import test_mysql_connection, get_db_connection, init_db
from matcher import SourceAFISMatcher


DATASET_DIR = CURRENT_DIR / "data" / "FAMILY FINGERPRINT DATASET"


def parse_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """
    Extract family, member, subject, impression, and file metadata from folder structure.
    Convention: .../FAMILY-<ID>/<MEMBER>/FM<ID>_<ROLE><IMPRESSION>.png
    """
    parts = file_path.resolve().parts
    filename = file_path.name
    stem = file_path.stem

    # Look backwards in path for FAMILY-X and MEMBER (CHILD/FATHER/MOTHER)
    family_id = None
    member_type = None

    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if p in ("CHILD", "FATHER", "MOTHER"):
            member_type = p
            if i > 0 and parts[i - 1].startswith("FAMILY-"):
                family_id = parts[i - 1]
            break

    if not family_id:
        match_fam = re.search(r"FAMILY-\d+", str(file_path))
        if match_fam:
            family_id = match_fam.group(0)

    if not member_type:
        for m in ("CHILD", "FATHER", "MOTHER"):
            if m in str(file_path).upper():
                member_type = m
                break

    # Extract impression ID from filename (e.g. FM001_C1 -> 1)
    impression_id = None
    match_imp = re.search(r"[CFM](\d+)$", stem, re.IGNORECASE)
    if match_imp:
        try:
            impression_id = int(match_imp.group(1))
        except ValueError:
            impression_id = None

    if not impression_id:
        match_any_digit = re.search(r"\d+$", stem)
        if match_any_digit:
            impression_id = int(match_any_digit.group(0))
        else:
            impression_id = 1

    subject_id = f"{family_id}/{member_type}" if family_id and member_type else (family_id or stem)

    return {
        "family_id": family_id or "UNKNOWN",
        "member_type": member_type or "UNKNOWN",
        "subject_id": subject_id,
        "impression_id": impression_id,
        "filename": filename,
        "image_path": str(file_path.relative_to(CURRENT_DIR)).replace("\\", "/"),
        "distortion_type": "clean",
        "dataset_split": "gallery",
    }


def process_single_image(file_path_str: str) -> Dict[str, Any]:
    """
    Worker function to validate image and generate SourceAFIS template.
    Runs in separate worker process.
    """
    file_path = Path(file_path_str)
    meta = parse_metadata_from_path(file_path)

    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return {
                "file_path": file_path_str,
                "valid": False,
                "error": "Corrupted or unreadable image file",
                "meta": meta,
                "template": None,
            }

        h, w = image.shape
        meta["image_width"] = int(w)
        meta["image_height"] = int(h)

        # Generate SourceAFIS template
        matcher = SourceAFISMatcher()
        template = matcher.extract_template(image)

        return {
            "file_path": file_path_str,
            "valid": True,
            "error": None,
            "meta": meta,
            "template": template,
        }

    except Exception as e:
        return {
            "file_path": file_path_str,
            "valid": False,
            "error": str(e),
            "meta": meta,
            "template": None,
        }


def import_family_dataset(
    dataset_dir: Optional[Path] = None,
    max_workers: int = 8,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main dataset import routine.
    Validates images, generates SourceAFIS templates, and inserts into MySQL.
    """
    print("=" * 60)
    print("FAMILY FINGERPRINT DATASET IMPORT & SOURCEAFIS INDEXING")
    print("=" * 60)

    # 1. Test MySQL Connection
    conn_ok, conn_msg = test_mysql_connection()
    if not conn_ok:
        print(f"MySQL connection: FAILED\nBlocker: {conn_msg}")
        return {"status": "error", "error": conn_msg}

    print("MySQL connection: SUCCESS")

    # 2. Ensure schema initialized
    init_db()

    target_dir = dataset_dir or DATASET_DIR
    if not target_dir.exists():
        err_msg = f"Dataset directory not found at: {target_dir}"
        print(f"ERROR: {err_msg}")
        return {"status": "error", "error": err_msg}

    # 3. Recursively find valid image files
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    discovered_files: List[Path] = []

    for root, _, files in os.walk(target_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in valid_exts:
                discovered_files.append(p)

    discovered_files.sort()
    total_discovered = len(discovered_files)
    print(f"Discovered {total_discovered} fingerprint image files.")

    if limit and limit > 0:
        discovered_files = discovered_files[:limit]
        print(f"Processing limit applied: {len(discovered_files)} images.")

    # 4. Check existing records in database to prevent duplicates
    db_conn = get_db_connection()
    cursor = db_conn.cursor(dictionary=True)
    cursor.execute("SELECT filename, distortion_type FROM fingerprints;")
    existing_keys = {
        (row["filename"], row["distortion_type"])
        for row in cursor.fetchall()
    }
    cursor.close()
    db_conn.close()

    print(f"Existing records in database: {len(existing_keys)}")

    files_to_process = [
        str(p) for p in discovered_files
        if (p.name, "clean") not in existing_keys
    ]
    duplicates_skipped = total_discovered - len(files_to_process)
    print(f"Files needing processing: {len(files_to_process)} (Duplicates to skip: {duplicates_skipped})")

    # 5. Process in parallel
    valid_count = 0
    invalid_count = 0
    inserted_count = 0
    templates_generated = 0
    templates_failed = 0
    invalid_images_report = []

    if files_to_process:
        print(f"Starting extraction with {max_workers} worker processes...")
        t_start = time.time()

        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        insert_fp_sql = """
            INSERT INTO fingerprints (
                subject_id, family_id, member_type, impression_id,
                filename, image_path, distortion_type, dataset_split,
                image_width, image_height
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        insert_tpl_sql = """
            INSERT INTO fingerprint_templates (
                fingerprint_id, template_data, minutiae_count,
                template_version, generation_status
            ) VALUES (%s, %s, %s, %s, %s);
        """

        processed_so_far = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, fpath): fpath
                for fpath in files_to_process
            }

            for future in as_completed(futures):
                res = future.result()
                processed_so_far += 1

                if not res["valid"]:
                    invalid_count += 1
                    invalid_images_report.append((res["file_path"], res["error"]))
                    continue

                valid_count += 1
                meta = res["meta"]
                template = res["template"]

                # Insert into fingerprints
                cursor.execute(
                    insert_fp_sql,
                    (
                        meta["subject_id"],
                        meta["family_id"],
                        meta["member_type"],
                        meta["impression_id"],
                        meta["filename"],
                        meta["image_path"],
                        meta["distortion_type"],
                        meta["dataset_split"],
                        meta["image_width"],
                        meta["image_height"],
                    ),
                )
                fingerprint_id = cursor.lastrowid
                inserted_count += 1

                # Insert into fingerprint_templates
                if template and template.get("minutiae_count", 0) > 0:
                    tpl_bytes = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)
                    cursor.execute(
                        insert_tpl_sql,
                        (
                            fingerprint_id,
                            tpl_bytes,
                            template["minutiae_count"],
                            template["version"],
                            "generated",
                        ),
                    )
                    templates_generated += 1
                else:
                    # Template generation failed or 0 minutiae
                    cursor.execute(
                        insert_tpl_sql,
                        (
                            fingerprint_id,
                            b"",
                            0,
                            "sourceafis_v1",
                            "failed",
                        ),
                    )
                    templates_failed += 1

                if processed_so_far % 50 == 0 or processed_so_far == len(files_to_process):
                    db_conn.commit()
                    elapsed = time.time() - t_start
                    rate = processed_so_far / max(1e-3, elapsed)
                    print(
                        f"[{processed_so_far}/{len(files_to_process)}] "
                        f"Imported: {inserted_count} | Templates: {templates_generated} | "
                        f"Rate: {rate:.1f} img/s | Elapsed: {elapsed:.1f}s"
                    )

        db_conn.commit()
        cursor.close()
        db_conn.close()

    print()
    print("=" * 60)
    print("DATASET IMPORT SUMMARY")
    print("=" * 60)
    print("MySQL connection: SUCCESS")
    print(f"Total images discovered: {total_discovered}")
    print(f"Valid images: {valid_count + (total_discovered - len(files_to_process)) if duplicates_skipped else valid_count}")
    print(f"Invalid images: {invalid_count}")
    print(f"Inserted records: {inserted_count}")
    print(f"Duplicates skipped: {duplicates_skipped}")
    print()
    print("SourceAFIS templates:")
    print(f"Generated: {templates_generated}")
    print(f"Failed: {templates_failed}")
    print("=" * 60)

    if invalid_images_report:
        print("\nInvalid image report:")
        for path, err in invalid_images_report[:10]:
            print(f"  {path}: {err}")

    return {
        "status": "success",
        "total_images_discovered": total_discovered,
        "valid_images": valid_count,
        "invalid_images": invalid_count,
        "inserted_records": inserted_count,
        "duplicates_skipped": duplicates_skipped,
        "templates_generated": templates_generated,
        "templates_failed": templates_failed,
    }


if __name__ == "__main__":
    import_family_dataset()
