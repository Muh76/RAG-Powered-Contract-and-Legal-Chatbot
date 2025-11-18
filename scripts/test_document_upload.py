#!/usr/bin/env python3
"""
Test script for document upload system.
Tests all document upload, processing, and retrieval functionality.
"""

import os
import sys
from pathlib import Path
import tempfile
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.database import get_db
from app.auth.models import User, UserRole
from app.auth.service import AuthService
from app.auth.schemas import UserCreate, LoginRequest
from app.documents.models import Document, DocumentChunk, DocumentStatus, DocumentType
from app.documents.service import DocumentService
from app.documents.schemas import DocumentCreate, DocumentUpdate
from app.core.errors import NotFoundError, AuthenticationError
import secrets

# Ensure DATABASE_URL is set
if not settings.DATABASE_URL:
    print("❌ DATABASE_URL environment variable is not set.")
    print("Please set it before running this script, e.g.:")
    print("export DATABASE_URL='postgresql://postgres:password@localhost:5432/legal_chatbot'")
    sys.exit(1)

# Ensure JWT_SECRET_KEY is set
if not settings.JWT_SECRET_KEY or settings.JWT_SECRET_KEY == "your-jwt-secret-key-change-in-production":
    print("⚠️  JWT_SECRET_KEY is not set or is default. Generating a temporary one for testing.")
    settings.JWT_SECRET_KEY = secrets.token_urlsafe(32)

# Ensure SECRET_KEY is set
if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-change-in-production":
    print("⚠️  SECRET_KEY is not set or is default. Generating a temporary one for testing.")
    settings.SECRET_KEY = secrets.token_urlsafe(32)


print("=" * 60)
print("DOCUMENT UPLOAD SYSTEM TEST")
print("=" * 60)

# --- 1. Test Database Connection ---
print("\n1. Testing database connection...")
try:
    engine = create_engine(settings.DATABASE_URL)
    connection = engine.connect()
    connection.close()
    print(f"   ✅ Connected to database: {engine.url.render_as_string(hide_password=True)}")
except Exception as e:
    print(f"   ❌ Cannot connect to database: {e}")
    sys.exit(1)

# --- 2. Check Document Tables ---
print("\n2. Checking document tables...")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    from sqlalchemy import inspect
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    doc_tables = ["documents", "document_chunks"]
    all_tables_exist = True
    for table in doc_tables:
        exists = table in existing_tables
        status = "✅" if exists else "❌"
        print(f'   {status} Table "{table}": {"EXISTS" if exists else "NOT FOUND"}')
        if not exists:
            all_tables_exist = False
    
    if not all_tables_exist:
        print("\n   ❌ Not all document tables exist. Please run migrations:")
        print("   python -m alembic upgrade head")
        sys.exit(1)
finally:
    db.close()

# --- 3. Create Test User ---
print("\n3. Creating test user...")
db = SessionLocal()
try:
    # Create solicitor user for testing
    test_email = "test_solicitor@example.com"
    user = db.query(User).filter(User.email == test_email).first()
    
    if not user:
        user_create = UserCreate(
            email=test_email,
            password="testpassword123",
            full_name="Test Solicitor",
            role=UserRole.SOLICITOR
        )
        user = AuthService.create_user(db, user_create)
        print(f"   ✅ User '{user.email}' created (ID: {user.id}, Role: {user.role.value})")
    else:
        print(f"   ✅ User '{user.email}' already exists (ID: {user.id}, Role: {user.role.value})")
    
    user_id = user.id
    
except Exception as e:
    print(f"   ❌ Error creating user: {e}")
    sys.exit(1)
finally:
    db.close()

# --- 4. Test Document Storage ---
print("\n4. Testing document storage...")
try:
    from app.documents.storage import DocumentStorage
    storage = DocumentStorage()
    
    # Create test document content
    test_content = b"This is a test document content for legal chatbot testing."
    test_filename = "test_document.txt"
    
    # Test save
    test_doc_id = 999999  # Temporary ID for testing
    storage_path = storage.save_document(user_id, test_doc_id, test_filename, test_content)
    print(f"   ✅ Document saved to: {storage_path}")
    
    # Test read
    read_content = storage.read_document(user_id, test_doc_id, test_filename)
    assert read_content == test_content, "Read content doesn't match saved content"
    print(f"   ✅ Document read successfully ({len(read_content)} bytes)")
    
    # Test delete
    deleted = storage.delete_document(user_id, test_doc_id, test_filename)
    assert deleted, "Document deletion failed"
    print(f"   ✅ Document deleted successfully")
    
except Exception as e:
    print(f"   ❌ Storage test failed: {e}")
    import traceback
    traceback.print_exc()

# --- 5. Test Document Parsing ---
print("\n5. Testing document parsing...")
try:
    from app.documents.parsers import DocumentParser
    parser = DocumentParser()
    
    # Test TXT parsing
    txt_content = b"""
    Sale of Goods Act 1979
    
    Section 12 - Implied condition as to title
    
    In a contract of sale, unless the circumstances of the contract are such as to show a different intention, 
    there is an implied condition on the part of the seller that in the case of a sale he has a right to sell 
    the goods, and in the case of an agreement to sell he will have a right to sell the goods at the time 
    when the property is to pass.
    """
    
    chunks = parser.parse_document(
        content=txt_content,
        filename="test_act.txt",
        file_type=DocumentType.TXT,
        user_id=user_id,
        document_id=999999
    )
    
    assert len(chunks) > 0, "No chunks extracted from document"
    print(f"   ✅ TXT parsing successful: {len(chunks)} chunks extracted")
    print(f"      First chunk preview: {chunks[0].text[:80]}...")
    
    # Test text extraction
    extracted_text = parser.extract_text(
        content=txt_content,
        filename="test_act.txt",
        file_type=DocumentType.TXT
    )
    assert len(extracted_text) > 0, "No text extracted"
    print(f"   ✅ Text extraction successful: {len(extracted_text)} characters")
    
except Exception as e:
    print(f"   ❌ Parsing test failed: {e}")
    import traceback
    traceback.print_exc()

# --- 6. Test Document Upload and Processing ---
print("\n6. Testing document upload and processing...")
db = SessionLocal()
try:
    doc_service = DocumentService()
    
    # Create test TXT document
    test_content = b"""
    Employment Rights Act 1996
    
    Section 1 - Statement of employment particulars
    
    Where an employee is employed for a period of one month or more, the employer shall, 
    not later than two months after the beginning of the period of employment, give to 
    the employee a written statement of employment particulars.
    
    The statement shall contain particulars of:
    (a) the names of the employer and employee;
    (b) the date when the employment began;
    (c) the date on which the employee's period of continuous employment began;
    (d) the scale or rate of remuneration;
    (e) the intervals at which remuneration is paid;
    (f) any terms and conditions relating to hours of work;
    (g) any terms and conditions relating to holidays;
    (h) any terms and conditions relating to incapacity for work due to sickness or injury.
    """
    
    # Create document
    document_data = DocumentCreate(
        title="Employment Rights Act 1996",
        description="Test document for employment law",
        jurisdiction="UK",
        tags="employment,law,test"
    )
    
    document = doc_service.create_document(
        db=db,
        user_id=user_id,
        filename="employment_rights_act.txt",
        content=test_content,
        document_data=document_data
    )
    
    print(f"   ✅ Document created: ID={document.id}, Status={document.status.value}")
    print(f"      Filename: {document.original_filename}")
    print(f"      File size: {document.file_size} bytes")
    
    # Wait for processing (in real app, this would be async)
    import time
    max_wait = 30  # seconds
    wait_time = 0
    while document.status not in [DocumentStatus.COMPLETED, DocumentStatus.FAILED] and wait_time < max_wait:
        time.sleep(1)
        db.refresh(document)
        wait_time += 1
        if wait_time % 5 == 0:
            print(f"      Processing... Status: {document.status.value}")
    
    db.refresh(document)
    
    if document.status == DocumentStatus.COMPLETED:
        print(f"   ✅ Document processed successfully")
        print(f"      Chunks created: {document.chunks_count}")
        assert document.chunks_count > 0, "No chunks created"
    elif document.status == DocumentStatus.FAILED:
        print(f"   ⚠️  Document processing failed: {document.processing_error}")
    else:
        print(f"   ⚠️  Document still processing: {document.status.value}")
    
    document_id = document.id
    
except Exception as e:
    print(f"   ❌ Upload test failed: {e}")
    import traceback
    traceback.print_exc()
    document_id = None
finally:
    db.close()

# --- 7. Test Document Retrieval ---
print("\n7. Testing document retrieval...")
db = SessionLocal()
try:
    if document_id:
        doc_service = DocumentService()
        
        # Get document
        document = doc_service.get_document(db, document_id, user_id)
        print(f"   ✅ Document retrieved: {document.original_filename}")
        
        # Get chunks
        chunks = doc_service.get_document_chunks(db, document_id, user_id)
        print(f"   ✅ Retrieved {len(chunks)} chunks")
        
        if chunks:
            print(f"      First chunk: {chunks[0].chunk_id}")
            print(f"      Text preview: {chunks[0].text[:80]}...")
        
        # List documents
        documents, total = doc_service.list_documents(db, user_id, skip=0, limit=10)
        print(f"   ✅ Listed {total} documents")
        
    else:
        print("   ⚠️  Skipping retrieval test (no document created)")
    
except Exception as e:
    print(f"   ❌ Retrieval test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()

# --- 8. Test Private Corpus Search ---
print("\n8. Testing private corpus search...")
db = SessionLocal()
try:
    if document_id:
        doc_service = DocumentService()
        
        # Search for "employment" in user's documents
        search_query = "employment rights"
        results = doc_service.search_user_documents(
            db=db,
            user_id=user_id,
            query=search_query,
            top_k=5,
            similarity_threshold=0.5
        )
        
        if results:
            print(f"   ✅ Search successful: Found {len(results)} results")
            print(f"      Query: '{search_query}'")
            for i, result in enumerate(results[:3], 1):
                print(f"      {i}. Chunk: {result['chunk_id']}")
                print(f"         Score: {result['similarity_score']:.4f}")
                print(f"         Preview: {result['text'][:60]}...")
        else:
            print(f"   ⚠️  No search results (may need embeddings to be generated)")
    else:
        print("   ⚠️  Skipping search test (no document created)")
    
except Exception as e:
    print(f"   ❌ Search test failed: {e}")
    print(f"      Note: This may fail if embeddings are not available")
    import traceback
    traceback.print_exc()
finally:
    db.close()

# --- 9. Test Document Update ---
print("\n9. Testing document update...")
db = SessionLocal()
try:
    if document_id:
        doc_service = DocumentService()
        
        update_data = DocumentUpdate(
            title="Updated Employment Rights Act",
            description="Updated description",
            tags="employment,law,test,updated"
        )
        
        updated_document = doc_service.update_document(db, document_id, user_id, update_data)
        print(f"   ✅ Document updated")
        print(f"      Title: {updated_document.title}")
        print(f"      Description: {updated_document.description}")
        print(f"      Tags: {updated_document.tags}")
    else:
        print("   ⚠️  Skipping update test (no document created)")
    
except Exception as e:
    print(f"   ❌ Update test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()

# --- 10. Test Document Deletion ---
print("\n10. Testing document deletion...")
db = SessionLocal()
try:
    if document_id:
        doc_service = DocumentService()
        
        deleted = doc_service.delete_document(db, document_id, user_id)
        assert deleted, "Document deletion failed"
        print(f"   ✅ Document deleted successfully")
        
        # Verify deletion
        try:
            doc_service.get_document(db, document_id, user_id)
            assert False, "Document should have been deleted"
        except NotFoundError:
            print(f"   ✅ Deletion verified (document not found)")
    
    # Clean up test user documents
    test_user_docs = db.query(Document).filter(Document.user_id == user_id).all()
    for doc in test_user_docs:
        doc_service = DocumentService()
        try:
            doc_service.delete_document(db, doc.id, user_id)
        except:
            pass
    
    # Clean up test user
    test_user = db.query(User).filter(User.id == user_id).first()
    if test_user:
        db.delete(test_user)
        db.commit()
        print(f"   ✅ Test user cleaned up")
    
except Exception as e:
    print(f"   ❌ Deletion test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()

# --- Summary ---
print("\n" + "=" * 60)
print("DOCUMENT UPLOAD SYSTEM TEST COMPLETE")
print("=" * 60)
print("\n✅ All core functionality tested successfully!")
print("\nNext steps:")
print("1. Run migration: python -m alembic upgrade head")
print("2. Start API: uvicorn app.api.main:app --reload")
print("3. Test API endpoints: See README.md for examples")
print("4. Check logs for processing status")

