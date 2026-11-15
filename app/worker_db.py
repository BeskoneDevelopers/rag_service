from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(settings.database_sync_url, echo=settings.debug)

sessionlocal = sessionmaker(bind=engine)
