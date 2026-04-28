from config.settings import SETTINGS
from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///" + str(SETTINGS.PATH_SQLITE_DB),
    connect_args={"check_same_thread": False},
)
