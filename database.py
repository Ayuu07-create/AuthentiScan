import sqlite3
from pathlib import Path


DATABASE_FOLDER = Path("instance")

DATABASE_PATH = (
    DATABASE_FOLDER /
    "forgery_detection.db"
)


def get_database_connection():
    DATABASE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def add_missing_analysis_columns(connection):
    """
    Add new columns to an existing database without
    deleting its current users or history.
    """

    column_rows = connection.execute(
        "PRAGMA table_info(analyses)"
    ).fetchall()

    existing_columns = {
        row["name"]
        for row in column_rows
    }

    required_columns = {
        "risk_score": "INTEGER",
        "quality_score": "INTEGER",
        "forensic_signals": "TEXT"
    }

    for column_name, column_type in (
        required_columns.items()
    ):
        if column_name not in existing_columns:
            connection.execute(
                f"""
                ALTER TABLE analyses
                ADD COLUMN {column_name} {column_type}
                """
            )


def initialize_database():
    connection = get_database_connection()

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            document_verified INTEGER NOT NULL,
            authenticity_score INTEGER,
            risk_score INTEGER,
            quality_score INTEGER,
            risk_level TEXT,
            forensic_signals TEXT,
            extracted_text TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
        );
        """
    )
def log_activity(
    user_id,
    action,
    details=None
):
    """
    Save one activity using a new database connection.
    """

    if user_id is None:
        return

    activity_connection = (
        get_database_connection()
    )

    try:
        activity_connection.execute(
            """
            INSERT INTO activity_logs (
                user_id,
                action,
                details
            )
            VALUES (?, ?, ?)
            """,
            (
                user_id,
                action,
                details
            )
        )

        activity_connection.commit()

    finally:
        activity_connection.close()