
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_NAME = Path(__file__).parent / "chat_history.db"


def create_table():
    """Create tables and safely migrate the existing database."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(chat_history)"
            ).fetchall()
        }

        if "sources" not in columns:
            conn.execute("""
                ALTER TABLE chat_history
                ADD COLUMN sources TEXT DEFAULT ''
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL UNIQUE,
                rating TEXT NOT NULL
                    CHECK (rating IN ('helpful', 'not_helpful')),
                comment TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id)
                    REFERENCES chat_history(id)
                    ON DELETE CASCADE
            )
        """)


def save_chat(question, answer, sources=""):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute("""
            INSERT INTO chat_history
                (question, answer, sources, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            question,
            answer,
            sources,
            datetime.now().astimezone().isoformat(timespec="seconds")
        ))
        return cursor.lastrowid


def get_chat_history(limit=100):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT id, question, answer, sources, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()


def delete_chat(chat_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "DELETE FROM chat_feedback WHERE chat_id = ?",
            (chat_id,)
        )
        conn.execute(
            "DELETE FROM chat_history WHERE id = ?",
            (chat_id,)
        )


def save_feedback(chat_id, rating, comment=""):
    if rating not in ("helpful", "not_helpful"):
        raise ValueError("Invalid feedback rating.")

    with sqlite3.connect(DB_NAME) as conn:
        # Only allow feedback for an existing chat.
        chat = conn.execute(
            "SELECT id FROM chat_history WHERE id = ?",
            (chat_id,)
        ).fetchone()

        if not chat:
            raise ValueError("Chat not found.")

        conn.execute("""
            INSERT INTO chat_feedback
                (chat_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                rating = excluded.rating,
                comment = excluded.comment,
                created_at = excluded.created_at
        """, (
            chat_id,
            rating,
            comment.strip(),
            datetime.now().astimezone().isoformat(timespec="seconds")
        ))


def get_feedback(chat_id):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT rating, comment, created_at
            FROM chat_feedback
            WHERE chat_id = ?
        """, (chat_id,)).fetchone()


def get_feedback_stats():
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*),
                COALESCE(SUM(
                    CASE WHEN rating = 'helpful' THEN 1 ELSE 0 END
                ), 0),
                COALESCE(SUM(
                    CASE WHEN rating = 'not_helpful' THEN 1 ELSE 0 END
                ), 0)
            FROM chat_feedback
        """).fetchone()

        return row


def get_total_questions():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT COUNT(*) FROM chat_history
        """).fetchone()[0]


def get_daily_usage(days=14):
    """Return daily question counts, including days with zero questions."""
    today = datetime.now().astimezone().date()
    start_date = today - timedelta(days=days - 1)

    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("""
            SELECT substr(created_at, 1, 10) AS day, COUNT(*)
            FROM chat_history
            WHERE substr(created_at, 1, 10) >= ?
              AND substr(created_at, 1, 10) <= ?
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day
        """, (
            start_date.isoformat(),
            today.isoformat()
        )).fetchall()

    counts = {day: count for day, count in rows}

    return [
        {
            "Date": (start_date + timedelta(days=i)).isoformat(),
            "Questions": counts.get(
                (start_date + timedelta(days=i)).isoformat(), 0
            )
        }
        for i in range(days)
    ]


def get_feedback_breakdown():
    total, helpful, not_helpful = get_feedback_stats()

    return [
        {"Rating": "Helpful", "Count": helpful},
        {"Rating": "Not Helpful", "Count": not_helpful},
    ]


def get_recent_feedback(limit=10):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT
                f.chat_id,
                h.question,
                f.rating,
                f.comment,
                f.created_at
            FROM chat_feedback AS f
            JOIN chat_history AS h ON h.id = f.chat_id
            ORDER BY f.created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()


def get_all_feedback(limit=100):
    return get_recent_feedback(limit=limit)


if __name__ == "__main__":
    create_table()
    print("Database ready. Existing chats and feedback preserved.")
    print("Total questions:", get_total_questions())
    print("Feedback totals:", get_feedback_stats())
