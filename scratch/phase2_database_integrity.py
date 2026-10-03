"""
Phase 2: Database Template Integrity, Uniqueness, and Mapping Check
Investigates:
1. Total fingerprint records in MySQL
2. Template generation status (generated, missing, failed)
3. SHA-256 hash uniqueness across all serialized templates
4. Table verification for specific candidate files:
   - FM001_F1.png
   - FM010_C1.png
   - FM074_M2.png
   - FM008_M2.png
   - FM028_C4.png
   - FM044_M3.png
5. Deserialization integrity: verify that stored template bytes deserialize back into valid minutiae arrays.
"""

import os
import sys
import pickle
import hashlib
import mysql.connector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database import get_db_connection

conn = get_db_connection()
cursor = conn.cursor(dictionary=True)

print("=" * 70)
print("PHASE 2: DATABASE TEMPLATE INTEGRITY & MAPPING AUDIT")
print("=" * 70)

# 1. Total records and breakdown
cursor.execute("SELECT COUNT(*) AS cnt FROM fingerprints")
total_prints = cursor.fetchone()["cnt"]

cursor.execute("""
    SELECT generation_status, COUNT(*) as cnt
    FROM fingerprint_templates
    GROUP BY generation_status
""")
status_counts = {row["generation_status"]: row["cnt"] for row in cursor.fetchall()}

cursor.execute("""
    SELECT COUNT(*) AS cnt
    FROM fingerprints f
    LEFT JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
    WHERE t.template_id IS NULL
""")
missing_templates = cursor.fetchone()["cnt"]

total_templates = sum(status_counts.values())
generated_cnt = status_counts.get("generated", 0)
failed_cnt = status_counts.get("failed", 0)

print(f"Total fingerprint records: {total_prints}")
print(f"Total templates in DB:     {total_templates}")
print(f"Templates generated:       {generated_cnt}")
print(f"Templates failed:          {failed_cnt}")
print(f"Templates missing:         {missing_templates}")

# 2. Template uniqueness test
cursor.execute("SELECT template_id, fingerprint_id, template_data FROM fingerprint_templates WHERE generation_status = 'generated'")
all_templates = cursor.fetchall()

hashes = {}
duplicates = []
corrupted = []
deserialized_counts = []

for row in all_templates:
    tid = row["template_id"]
    blob = row["template_data"]
    h = hashlib.sha256(blob).hexdigest()
    if h in hashes:
        duplicates.append((tid, hashes[h], h))
    else:
        hashes[h] = tid

    # Try deserializing
    try:
        data = pickle.loads(blob)
        if "minutiae" not in data or not hasattr(data["minutiae"], "shape"):
            corrupted.append(tid)
        else:
            deserialized_counts.append(data["minutiae"].shape[0])
    except Exception as e:
        corrupted.append(tid)

print(f"\n--- TEMPLATE UNIQUENESS & INTEGRITY ---")
print(f"Total evaluated templates:      {len(all_templates)}")
print(f"Unique serialized templates:    {len(hashes)}")
print(f"Duplicate serialized templates: {len(duplicates)}")
print(f"Corrupted / invalid templates:  {len(corrupted)}")
dup_pct = (len(duplicates) / len(all_templates) * 100.0) if all_templates else 0.0
print(f"Percentage duplicated:          {dup_pct:.2f}%")

if deserialized_counts:
    min_m = min(deserialized_counts)
    max_m = max(deserialized_counts)
    avg_m = sum(deserialized_counts) / len(deserialized_counts)
    print(f"Minutiae per template - Min: {min_m}, Max: {max_m}, Mean: {avg_m:.1f}")

# 3. Specific Candidate Mapping
target_files = [
    "FM001_F1.png",
    "FM010_C1.png",
    "FM074_M2.png",
    "FM008_M2.png",
    "FM028_C4.png",
    "FM044_M3.png",
]

print(f"\n--- SPECIFIC CANDIDATE MAPPING TABLE ---")
print(f"{'Filename':<14} | {'Family':<10} | {'Member':<8} | {'FP ID':<6} | {'Tpl ID':<6} | {'Minutiae':<8} | {'Tpl Hash (first 16)':<20}")
print("-" * 85)

for fname in target_files:
    cursor.execute("""
        SELECT f.fingerprint_id, f.filename, f.family_id, f.member_type, f.image_path,
               t.template_id, t.template_data, t.minutiae_count
        FROM fingerprints f
        LEFT JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
        WHERE f.filename = %s
    """, (fname,))
    row = cursor.fetchone()
    if not row:
        print(f"{fname:<14} | NOT FOUND IN DATABASE")
        continue

    blob = row["template_data"]
    thash = hashlib.sha256(blob).hexdigest()[:16] if blob else "NULL"
    print(f"{row['filename']:<14} | {row['family_id']:<10} | {row['member_type']:<8} | {row['fingerprint_id']:<6} | {str(row['template_id']):<6} | {str(row['minutiae_count']):<8} | {thash:<20}")

cursor.close()
conn.close()
print("=" * 70)
