import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/test")
os.environ.setdefault("DATABASE_SYNC_URL", "postgresql+psycopg://localhost/test")
os.environ.setdefault("DEBUG", "false")
