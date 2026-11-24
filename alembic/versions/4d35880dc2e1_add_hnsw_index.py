"""add_hnsw_index

Revision ID: 4d35880dc2e1
Revises: 22b5d132faa8
Create Date: 2026-11-15 17:17:50.756677

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4d35880dc2e1"
down_revision: Union[str, Sequence[str], None] = "22b5d132faa8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.execute("""
        CREATE INDEX ix_chunk_embedding_hnsw
        ON chunk
        USING hnsw (embedding vector_cosine_ops);
    """)


def downgrade() -> None:
    """Downgrade schema."""

    op.execute("""
            DROP INDEX IF EXISTS ix_chunk_embedding_hnsw;
        """)
