"""
Load extraction/extracted_database.json into Postgres.

    python seed_db.py            add new questions (duplicates are skipped)
    python seed_db.py --reset    empty the Questions table first
"""
import argparse
import json
import os
import uuid
from pathlib import Path

import psycopg2

DATA_FILE = Path(__file__).resolve().parent.parent / "extraction" / "extracted_database.json"

DB = dict(
    dbname=os.getenv("JEE_DB_NAME", "jeedb"),
    user=os.getenv("JEE_DB_USER", "jeeadmin"),
    password=os.getenv("JEE_DB_PASSWORD", "your_secure_password"),  # or set JEE_DB_PASSWORD
    host=os.getenv("JEE_DB_HOST", "localhost"),
)

# Safe to run repeatedly; also upgrades a table created by an older version.
SCHEMA = """ CREATE TABLE IF NOT EXISTS Questions (
    id UUID PRIMARY KEY,
    subject TEXT NOT NULL,
    topic TEXT,
    complexity TEXT,
    question_text TEXT NOT NULL,
    option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
    correct_option CHAR(1)
);
ALTER TABLE Questions ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE Questions ADD COLUMN IF NOT EXISTS complexity TEXT;
ALTER TABLE Questions ALTER COLUMN correct_option DROP NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS questions_text_uniq ON Questions (md5(question_text));
"""


def norm_subject(s):
    s = str(s or "").strip().lower()
    if s.startswith("phys"):
        return "Physics"
    if s.startswith("chem"):
        return "Chemistry"
    if s.startswith("math"):  # Math / Maths / Mathematics
        return "Math"
    return None


def norm_level(s):
    s = str(s or "").strip().capitalize()
    return s if s in ("Easy", "Medium", "Hard") else "Medium"


def norm_key(s):
    s = str(s or "").strip().upper()[:1]
    return s if s in ("A", "B", "C", "D") else None  # NULL, never a made-up 'A'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="TRUNCATE Questions before loading")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Not found: {DATA_FILE}\nRun extraction/extract.py first.")
    rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    inserted = duplicates = invalid = unkeyed = 0
    conn = psycopg2.connect(**DB)
    try:
        with conn, conn.cursor() as cur:  # commits on success, rolls back on error
            cur.execute(SCHEMA)
            if args.reset:
                cur.execute("TRUNCATE Questions")

            for q in rows:
                if not isinstance(q, dict):
                    invalid += 1
                    continue
                subject = norm_subject(q.get("subject"))
                needed = ("question_text", "option_a", "option_b", "option_c", "option_d")
                if not subject or not all(str(q.get(k) or "").strip() for k in needed):
                    invalid += 1
                    continue

                key = norm_key(q.get("correct_option"))
                cur.execute(
                    """
                    INSERT INTO Questions
                        (id, subject, topic, complexity, question_text,
                         option_a, option_b, option_c, option_d, correct_option)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()), subject,
                        (q.get("chapter") or "").strip() or "Uncategorised",
                        norm_level(q.get("complexity")),
                        q["question_text"].strip(),
                        q["option_a"], q["option_b"], q["option_c"], q["option_d"],
                        key,
                    ),
                )
                if cur.rowcount:
                    inserted += 1
                    unkeyed += key is None
                else:
                    duplicates += 1
    finally:
        conn.close()

    print(f"Inserted {inserted} | duplicates skipped {duplicates} | invalid rows skipped {invalid}")
    if unkeyed:
        print(f"{unkeyed} inserted questions have no answer key yet - set them in the admin dashboard "
              f"(they are excluded from exams until then).")


if __name__ == "__main__":
    main()
