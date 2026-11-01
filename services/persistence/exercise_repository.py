import sqlite3
import streamlit as st
from pathlib import Path

_DB_PATH = str(Path(__file__).parent.parent.parent / "data.db")


@st.cache_resource
def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _fix_created_at(conn, table):
    """Repairs old tables where created_at was misspelled (created_id) or missing."""
    columns = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]

    if "created_at" not in columns:
        if "created_id" in columns:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN created_id TO created_at")
        else:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN created_at TIMESTAMP")

    conn.execute(f"UPDATE {table} SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")


def init_db() -> None:
    conn = _get_connection()

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT    NOT NULL,
                reps          INTEGER NOT NULL DEFAULT 0,
                sets          INTEGER NOT NULL DEFAULT 0,
                time          INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        _fix_created_at(conn, "users")
        _fix_created_at(conn, "exercises")


def get_user(username: str) -> sqlite3.Row:
    conn = _get_connection()

    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def create_user(username: str) -> sqlite3.Row:
    conn = _get_connection()

    with conn:
        conn.execute(
            "INSERT INTO users (username, created_at) VALUES (?, CURRENT_TIMESTAMP)",
            (username,),
        )

    return get_user(username)


def get_or_create_user(username: str) -> sqlite3.Row:
    user = get_user(username)

    if user is None:
        user = create_user(username)

    return user


def add_exercise(user_id, exercise_name, reps, sets, time):
    """Always saves a NEW history row. Rows are never merged."""
    conn = _get_connection()

    with conn:
        conn.execute("""
            INSERT INTO exercises (user_id, exercise_name, sets, reps, time, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, exercise_name, sets, reps, time))


def get_users_exercises(user_id):
    conn = _get_connection()

    return conn.execute("""
        SELECT * FROM exercises
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,)).fetchall()