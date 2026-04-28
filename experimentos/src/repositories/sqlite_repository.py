from loguru import logger
from sqlalchemy import text

from src.db.sqlite import get_engine


def execute_query(query: str) -> list[dict]:
    """Executes a raw SQL query and returns results as list of dicts."""
    try:
        with get_engine().connect() as connection:
            result = connection.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise
