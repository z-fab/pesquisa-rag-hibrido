from sqlalchemy import create_engine

from src.config.settings import SETTINGS

_engine = None


def get_engine():
    """Returns the SQLAlchemy engine, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            "sqlite:///" + str(SETTINGS.PATH_SQLITE_DB),
            connect_args={"check_same_thread": False},
        )
    return _engine
