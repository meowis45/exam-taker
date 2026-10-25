"""
JEE platform backend.

Run from the backend/ folder:   uvicorn main:app --reload
Admin API : http://localhost:8000/api/admin/questions
Exam page : http://localhost:8000/exam/        (served from "exam frontend/Jee-Simulation")
"""
import os
import random
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
EXAM_DIR = ROOT / "exam frontend" / "Jee-Simulation"

DB = dict(
    dbname=os.getenv("JEE_DB_NAME", "jeedb"),
    user=os.getenv("JEE_DB_USER", "jeeadmin"),
    password=os.getenv("JEE_DB_PASSWORD", "your_secure_password"),  # or set JEE_DB_PASSWORD
    host=os.getenv("JEE_DB_HOST", "localhost"),
)

LEVELS = ["Easy", "Medium", "Hard"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def db_cursor(commit: bool = False):
    """Always closes the connection, even if the query raises."""
    conn = psycopg2.connect(**DB)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


@app.exception_handler(psycopg2.Error)
async def db_error_handler(request, exc):
    # Returned as JSON (with CORS headers) so the frontend can show the real reason,
    # e.g. 'relation "questions" does not exist' -> run seed_db.py.
    msg = (exc.pgerror or str(exc)).strip()
    return JSONResponse(status_code=503, content={"detail": f"Database error: {msg}"})


# --------------------------------------------------------------------------
# 1. ADMIN ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/api/admin/questions")
def get_all_questions():
    """Reads from Postgres (the same source the exam uses), not the raw JSON file."""
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT id::text AS id, subject, topic AS chapter, complexity, question_text,
                   option_a, option_b, option_c, option_d, correct_option
            FROM Questions
            ORDER BY subject, topic, question_text
            """
        )
        rows = cur.fetchall()
    return {"total": len(rows), "questions": rows}


class QuestionUpdate(BaseModel):
    complexity: Optional[Literal["Easy", "Medium", "Hard"]] = None
    correct_option: Optional[Literal["A", "B", "C", "D"]] = None


@app.patch("/api/admin/questions/{question_id}")
def update_question(question_id: str, body: QuestionUpdate):
    """Lets the admin dashboard set the answer key and fix the difficulty."""
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to update")
    sets = ", ".join(f"{col} = %s" for col in fields)  # column names come from the model, not the user
    with db_cursor(commit=True) as cur:
        cur.execute(f"UPDATE Questions SET {sets} WHERE id = %s", [*fields.values(), question_id])
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Question not found")
    return {"ok": True}


@app.get("/api/meta")
def get_meta():
    """Subjects/chapters available for building an exam. 'keyed' = has an answer key."""
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT subject, topic, COUNT(*) AS total, COUNT(correct_option) AS keyed
            FROM Questions
            GROUP BY subject, topic
            ORDER BY subject, topic
            """
        )
        return {"topics": cur.fetchall()}


# --------------------------------------------------------------------------
# 2. EXAM GENERATOR
# --------------------------------------------------------------------------
class ExamRequest(BaseModel):
    subjects: list[str] = Field(min_length=1)
    chapters: Optional[list[str]] = None  # None / empty = all chapters
    difficulty_mode: Literal["Easy", "Medium", "Hard"] = "Medium"
    questions_per_subject: int = Field(30, ge=1, le=60)


# Mix per 30 questions (same numbers as before); scaled to questions_per_subject.
BASE_QUOTA = {
    "Hard": {"Hard": 15, "Medium": 9, "Easy": 6},
    "Medium": {"Hard": 6, "Medium": 14, "Easy": 10},
    "Easy": {"Hard": 2, "Medium": 12, "Easy": 16},
}


def quota_for(mode: str, n: int) -> dict:
    quota = {lvl: round(cnt * n / 30) for lvl, cnt in BASE_QUOTA[mode].items()}
    quota["Medium"] = max(0, quota["Medium"] + n - sum(quota.values()))
    return quota


@app.post("/generate_exam")
def generate_exam(request: ExamRequest):
    n = request.questions_per_subject
    quota = quota_for(request.difficulty_mode, n)
    exam_paper, warnings = [], []

    with db_cursor() as cur:
        for subject in request.subjects:
            # Only questions that have an answer key can be scored.
            sql = """
                SELECT id::text AS id, subject, topic AS chapter, complexity, question_text,
                       option_a, option_b, option_c, option_d, correct_option
                FROM Questions
                WHERE subject = %s AND correct_option IS NOT NULL
            """
            params = [subject]
            if request.chapters:
                sql += " AND topic = ANY(%s)"
                params.append(request.chapters)
            sql += " ORDER BY RANDOM()"
            cur.execute(sql, params)
            rows = cur.fetchall()

            pools = {lvl: [r for r in rows if r["complexity"] == lvl] for lvl in LEVELS}
            picked = []
            for lvl, count in quota.items():
                picked += pools[lvl][:count]

            # If a difficulty bucket was short, top up from whatever is left.
            chosen = {r["id"] for r in picked}
            picked += [r for r in rows if r["id"] not in chosen][: n - len(picked)]
            picked = picked[:n]

            if len(picked) < n:
                warnings.append(f"{subject}: only {len(picked)} of {n} questions available")
            exam_paper.extend(picked)

    if not exam_paper:
        raise HTTPException(
            status_code=404,
            detail="No questions with an answer key match this selection. "
                   "Set answers in the admin dashboard first.",
        )

    random.shuffle(exam_paper)
    return {
        "exam_id": str(uuid.uuid4()),
        "duration": max(1, round(len(exam_paper) * 2)),  # minutes (2 per question, as before: 60 per 30)
        "total_questions": len(exam_paper),
        "warnings": warnings,
        "questions": exam_paper,
    }


# Serves the exam page at http://localhost:8000/exam/ so it never runs from file://
if EXAM_DIR.exists():
    app.mount("/exam", StaticFiles(directory=EXAM_DIR, html=True), name="exam")
