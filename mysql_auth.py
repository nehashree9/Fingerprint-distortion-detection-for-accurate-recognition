"""
MySQL Database Authentication & Template Repository for Fingerprint Recognition.
Queries fingerprints and SourceAFIS templates from normalized schema in fingerprint_db.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Dict, List, Optional

import mysql.connector

from database import get_db_connection, MATCH_THRESHOLD

# In-memory cache for fast 1:N searching across gallery templates
_GALLERY_CACHE: Optional[List[Dict[str, Any]]] = None
_LAST_CACHE_TIME: float = 0.0
CACHE_TTL_SECONDS = 300.0


def get_person_templates(subject_id: str, mysql_password: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve all enrolled fingerprint templates for a given subject (e.g. 'FAMILY-1/CHILD').
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT
            f.fingerprint_id,
            f.subject_id,
            f.family_id,
            f.member_type,
            f.impression_id,
            f.filename,
            f.image_path,
            f.image_width,
            f.image_height,
            t.template_id,
            t.template_data,
            t.minutiae_count,
            t.template_version,
            t.generation_status
        FROM fingerprints f
        JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
        WHERE (f.subject_id = %s OR f.family_id = %s)
          AND t.generation_status = 'generated'
        ORDER BY f.impression_id;
        """,
        (subject_id, subject_id),
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise ValueError(f"No enrolled fingerprint templates found for subject: {subject_id}")

    templates = []
    for row in rows:
        try:
            tpl_data = pickle.loads(row["template_data"])
        except Exception:
            tpl_data = None

        if tpl_data is not None:
            templates.append(
                {
                    "fingerprint_id": row["fingerprint_id"],
                    "template_id": row["template_id"],
                    "impression_id": row["impression_id"],
                    "filename": row["filename"],
                    "image_path": row["image_path"],
                    "template": tpl_data,
                    "minutiae_count": row["minutiae_count"],
                    "image_width": row["image_width"],
                    "image_height": row["image_height"],
                }
            )

    return {
        "subject_id": rows[0]["subject_id"],
        "family_id": rows[0]["family_id"],
        "member_type": rows[0]["member_type"],
        "templates": templates,
    }


def get_all_gallery_templates(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve all enrolled gallery templates from MySQL for 1:N identification.
    Uses in-memory cache to guarantee sub-second search times.
    """
    global _GALLERY_CACHE, _LAST_CACHE_TIME

    now = time.time()
    if not force_refresh and _GALLERY_CACHE is not None and (now - _LAST_CACHE_TIME < CACHE_TTL_SECONDS):
        return _GALLERY_CACHE

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT
            f.fingerprint_id,
            f.subject_id,
            f.family_id,
            f.member_type,
            f.impression_id,
            f.filename,
            f.image_path,
            f.image_width,
            f.image_height,
            t.template_id,
            t.template_data,
            t.minutiae_count
        FROM fingerprints f
        JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
        WHERE t.generation_status = 'generated'
        ORDER BY f.family_id, f.member_type, f.impression_id;
        """
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    gallery = []
    for row in rows:
        try:
            tpl_data = pickle.loads(row["template_data"])
        except Exception:
            continue

        if tpl_data is not None and tpl_data.get("minutiae_count", 0) > 0:
            gallery.append(
                {
                    "fingerprint_id": row["fingerprint_id"],
                    "subject_id": row["subject_id"],
                    "family_id": row["family_id"],
                    "member_type": row["member_type"],
                    "impression_id": row["impression_id"],
                    "filename": row["filename"],
                    "image_path": row["image_path"],
                    "minutiae_count": row["minutiae_count"],
                    "image_width": row["image_width"],
                    "image_height": row["image_height"],
                    "template": tpl_data,
                }
            )

    _GALLERY_CACHE = gallery
    _LAST_CACHE_TIME = now
    return gallery


def clear_gallery_cache():
    """Clear in-memory gallery cache."""
    global _GALLERY_CACHE, _LAST_CACHE_TIME
    _GALLERY_CACHE = None
    _LAST_CACHE_TIME = 0.0