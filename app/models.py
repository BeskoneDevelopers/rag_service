from pgvector.sqlalchemy import VECTOR
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(index=True)
    company: Mapped[str] = mapped_column(index=True)
    file_path: Mapped[str]
    content_hash: Mapped[str]
    status: Mapped[str]
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete, delete-orphan", passive_deletes=True
    )

    __table_args__ = (UniqueConstraint("company", "content_hash", name="uq_document_company_hash"),)


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]
    content: Mapped[str]
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    section: Mapped[str | None] = mapped_column(nullable=True)
    company: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(VECTOR(1024))
    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
    )
