#!/usr/bin/env python3
"""
Test Authentication Database Setup
Verifies database tables, models, and authentication functionality
"""

import os
import sys
from pathlib import Path

# Set environment variables
if not os.environ.get('DATABASE_URL'):
    print("Set DATABASE_URL environment variable to run tests.")
    sys.exit(1)
if not os.environ.get('JWT_SECRET_KEY') and not os.environ.get('JWT_SECRET'):
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-testing-only'
if not os.environ.get('SECRET_KEY'):
    os.environ['SECRET_KEY'] = 'test-secret-key-for-testing-only'

print('='*60)
print('AUTHENTICATION DATABASE TEST')
print('='*60)

try:
    # Test 1: Database connection
    print('\n1. Testing database connection...')
    import psycopg2
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        cursor.execute('SELECT version()')
        version = cursor.fetchone()[0]
        print(f'   ✅ Connected to database: {version[:50]}...')
        conn.close()
    except psycopg2.OperationalError as e:
        print(f'   ❌ Cannot connect to database: {e}')
        print('\n   Please ensure PostgreSQL is running:')
        print('   - Docker: docker compose up -d postgres')
        print('   - Homebrew: brew services start postgresql@15')
        sys.exit(1)
    
    # Test 2: Check tables exist
    print('\n2. Checking authentication tables...')
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cursor = conn.cursor()
    
    tables_to_check = ['users', 'oauth_accounts', 'refresh_tokens']
    all_exist = True
    
    for table in tables_to_check:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """, (table,))
        exists = cursor.fetchone()[0]
        status = '✅' if exists else '❌'
        exists_str = 'EXISTS' if exists else 'NOT FOUND'
        print(f'   {status} Table "{table}": {exists_str}')
        if not exists:
            all_exist = False
    
    if not all_exist:
        print('\n   ❌ Some tables are missing. Please run migrations:')
        print('   python -m alembic upgrade head')
        sys.exit(1)
    
    conn.close()
    
    # Test 3: Test SQLAlchemy models
    print('\n3. Testing SQLAlchemy models...')
    from app.core.database import SessionLocal
    from app.auth.models import User, OAuthAccount, RefreshToken, UserRole
    
    db = SessionLocal()
    try:
        # Test creating a user
        test_user = User(
            email='test_db@example.com',
            hashed_password='test_hash',
            full_name='Test User',
            role=UserRole.PUBLIC
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        print(f'   ✅ User model works: Created user ID={test_user.id}')
        
        # Test querying
        found_user = db.query(User).filter(User.email == 'test_db@example.com').first()
        if found_user:
            print(f'   ✅ User query works: Found user ID={found_user.id}')
        
        # Clean up
        db.delete(test_user)
        db.commit()
        print('   ✅ Test data cleaned up')
        
    except Exception as e:
        print(f'   ❌ Model test failed: {e}')
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()
    
    # Test 4: Test authentication service
    print('\n4. Testing authentication service...')
    from app.auth.service import AuthService
    from app.auth.schemas import UserCreate
    
    db = SessionLocal()
    try:
        # Create user
        user_data = UserCreate(
            email='service_test@example.com',
            password='testpass123',
            full_name='Service Test User',
            role=UserRole.PUBLIC
        )
        
        # Check if exists
        existing = db.query(User).filter(User.email == user_data.email).first()
        if existing:
            from app.auth.models import RefreshToken
            db.query(RefreshToken).filter(RefreshToken.user_id == existing.id).delete()
            db.delete(existing)
            db.commit()
        
        user = AuthService.create_user(db, user_data)
        print(f'   ✅ User registration works: ID={user.id}, Email={user.email}')
        
        # Test authentication
        authenticated = AuthService.authenticate_user(db, 'service_test@example.com', 'testpass123')
        if authenticated:
            print(f'   ✅ User authentication works: User ID={authenticated.id}')
        else:
            print('   ❌ Authentication failed')
            sys.exit(1)
        
        # Test token creation
        tokens = AuthService.create_tokens(user, db)
        print(f'   ✅ Token creation works: access_token={len(tokens["access_token"])} chars')
        
        # Clean up
        from app.auth.models import RefreshToken
        db.query(RefreshToken).filter(RefreshToken.user_id == user.id).delete()
        db.delete(user)
        db.commit()
        print('   ✅ Test data cleaned up')
        
    except Exception as e:
        print(f'   ❌ Service test failed: {e}')
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()
    
    print('\n' + '='*60)
    print('✅ ALL DATABASE TESTS PASSED')
    print('='*60)
    print('\nDatabase setup is complete and working correctly!')
    print('\nNext steps:')
    print('  1. Protect existing routes with authentication')
    print('  2. Test API endpoints')
    print('  3. Implement document upload system (Phase 5.3)')
    
except ImportError as e:
    print(f'\n❌ Import error: {e}')
    print('Please install dependencies: pip install -r requirements.txt')
    sys.exit(1)
except Exception as e:
    print(f'\n❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

