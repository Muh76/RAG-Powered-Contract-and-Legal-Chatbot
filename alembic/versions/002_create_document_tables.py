"""create document tables

Revision ID: 002
Revises: 001
Create Date: 2024-11-18 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create ENUM types for document status and type
    op.execute("CREATE TYPE documentstatus AS ENUM ('uploaded', 'parsing', 'processing', 'indexing', 'completed', 'failed')")
    op.execute("CREATE TYPE documenttype AS ENUM ('pdf', 'docx', 'txt', 'md')")
    
    # Create documents table
    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=False),
    sa.Column('original_filename', sa.String(length=255), nullable=False),
    sa.Column('file_type', postgresql.ENUM('pdf', 'docx', 'txt', 'md', name='documenttype', create_type=False), nullable=False),
    sa.Column('file_size', sa.Integer(), nullable=False),
    sa.Column('file_path', sa.String(length=500), nullable=False),
    sa.Column('storage_path', sa.String(length=500), nullable=True),
    sa.Column('title', sa.String(length=255), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('extracted_text', sa.Text(), nullable=True),
    sa.Column('status', postgresql.ENUM('uploaded', 'parsing', 'processing', 'indexing', 'completed', 'failed', name='documentstatus', create_type=False), nullable=False),
    sa.Column('chunks_count', sa.Integer(), nullable=True),
    sa.Column('processing_error', sa.Text(), nullable=True),
    sa.Column('additional_metadata', sa.Text(), nullable=True),
    sa.Column('jurisdiction', sa.String(length=50), nullable=True),
    sa.Column('tags', sa.String(length=500), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('processed_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
    op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
    
    # Create document_chunks table
    op.create_table('document_chunks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('chunk_id', sa.String(length=255), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('start_char', sa.Integer(), nullable=False),
    sa.Column('end_char', sa.Integer(), nullable=False),
    sa.Column('embedding', sa.Text(), nullable=True),
    sa.Column('embedding_dimension', sa.Integer(), nullable=True),
    sa.Column('section', sa.String(length=255), nullable=True),
    sa.Column('title', sa.String(length=255), nullable=True),
    sa.Column('additional_metadata', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('chunk_id')
    )
    op.create_index(op.f('ix_document_chunks_id'), 'document_chunks', ['id'], unique=False)
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_user_id'), 'document_chunks', ['user_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_chunk_id'), 'document_chunks', ['chunk_id'], unique=True)


def downgrade():
    op.drop_index(op.f('ix_document_chunks_chunk_id'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_user_id'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_id'), table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_id'), table_name='documents')
    op.drop_table('documents')
    op.execute("DROP TYPE documentstatus")
    op.execute("DROP TYPE documenttype")

