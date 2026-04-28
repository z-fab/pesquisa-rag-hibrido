from db.sqlite import engine
from loguru import logger
from sqlalchemy import text


def execute_query(query: str) -> list[dict]:
    """Executes a raw SQL query against the SQLite database and returns the results."""

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []
