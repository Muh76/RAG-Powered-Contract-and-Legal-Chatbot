#!/usr/bin/env python3
"""
Test API Endpoints with Authentication
Tests protected endpoints using HTTP requests
"""
import os
import sys
import requests
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgresql://javadbeni@localhost:5432/legal_chatbot')
os.environ['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'test-secret-key-for-testing')
os.environ['SECRET_KEY'] = os.getenv('SECRET_KEY', 'test-secret-key')

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

print('='*60)
print('API ENDPOINT PROTECTION TEST')
print('='*60)
print(f'API URL: {API_BASE_URL}')
print('='*60)

try:
    # Test 1: Create test users and get tokens
    print('\n1. Creating test users and tokens...')
    from app.core.database import SessionLocal
    from app.auth.service import AuthService
    from app.auth.schemas import UserCreate, UserRole
    from app.auth.models import User, RefreshToken
    
    db = SessionLocal()
    
    test_users = {}
    test_tokens = {}
    
    try:
        # Create users
        users_data = {
            'public': UserCreate(
                email='api_test_public@example.com',
                password='testpass123',
                full_name='API Test Public',
                role=UserRole.PUBLIC
            ),
            'solicitor': UserCreate(
                email='api_test_solicitor@example.com',
                password='testpass123',
                full_name='API Test Solicitor',
                role=UserRole.SOLICITOR
            ),
            'admin': UserCreate(
                email='api_test_admin@example.com',
                password='testpass123',
                full_name='API Test Admin',
                role=UserRole.ADMIN
            ),
        }
        
        for role_name, user_data in users_data.items():
            # Clean up existing
            existing = db.query(User).filter(User.email == user_data.email).first()
            if existing:
                db.query(RefreshToken).filter(RefreshToken.user_id == existing.id).delete()
                db.delete(existing)
                db.commit()
            
            # Create user
            user = AuthService.create_user(db, user_data)
            tokens = AuthService.create_tokens(user, db)
            test_users[role_name] = user
            test_tokens[role_name] = tokens
            
            print(f'   ✅ Created {role_name} user: {user.email}')
            print(f'      Access token: {tokens["access_token"][:50]}...')
        
    finally:
        db.close()
    
    # Test 2: Test public endpoint (health) - should work without auth
    print('\n2. Testing Public Endpoint (Health)...')
    try:
        response = requests.get(f'{API_BASE_URL}/api/v1/health', timeout=5)
        if response.status_code == 200:
            print('   ✅ Health endpoint accessible without authentication')
        else:
            print(f'   ⚠️  Health endpoint returned {response.status_code} (expected 200)')
    except requests.exceptions.ConnectionError:
        print('   ⚠️  API server not running - skipping HTTP tests')
        print('   (Start server with: uvicorn app.api.main:app --reload)')
        api_server_running = False
    except Exception as e:
        print(f'   ⚠️  Health endpoint test error: {e}')
        api_server_running = False
    else:
        api_server_running = True
    
    if api_server_running:
        # Test 3: Test protected endpoints without authentication
        print('\n3. Testing Protected Endpoints WITHOUT Authentication...')
        protected_endpoints = [
            ('POST', '/api/v1/chat', {'query': 'test query'}),
            ('POST', '/api/v1/search/hybrid', {'query': 'test', 'top_k': 5}),
            ('GET', '/api/v1/documents', None),
            ('POST', '/api/v1/agentic-chat', {'query': 'test', 'mode': 'public'}),
            ('GET', '/api/v1/metrics', None),
        ]
        
        for method, endpoint, data in protected_endpoints:
            try:
                url = f'{API_BASE_URL}{endpoint}'
                if method == 'POST':
                    response = requests.post(url, json=data, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                
                if response.status_code in [401, 403]:
                    print(f'   ✅ {method} {endpoint}: Correctly requires authentication (status {response.status_code})')
                else:
                    print(f'   ⚠️  {method} {endpoint}: Unexpected status {response.status_code} (expected 401/403)')
            except requests.exceptions.ConnectionError:
                print(f'   ⚠️  Server not running - skipping {endpoint}')
            except Exception as e:
                print(f'   ⚠️  Error testing {endpoint}: {e}')
        
        # Test 4: Test protected endpoints with authentication
        print('\n4. Testing Protected Endpoints WITH Authentication...')
        
        # Test chat endpoint (all authenticated users)
        for role in ['public', 'solicitor', 'admin']:
            token = test_tokens[role]['access_token']
            try:
                response = requests.post(
                    f'{API_BASE_URL}/api/v1/chat',
                    json={'query': 'test query'},
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5
                )
                if response.status_code not in [401, 403]:
                    print(f'   ✅ Chat endpoint accessible with {role} token (status {response.status_code})')
                else:
                    print(f'   ⚠️  Chat endpoint denied for {role} (status {response.status_code})')
            except Exception as e:
                print(f'   ⚠️  Error testing chat with {role} token: {e}')
        
        # Test 5: Test RBAC endpoints
        print('\n5. Testing RBAC-Protected Endpoints...')
        
        # Document upload (Solicitor/Admin only)
        for role in ['public', 'solicitor', 'admin']:
            token = test_tokens[role]['access_token']
            try:
                # Create a dummy file
                files = {'file': ('test.pdf', b'fake pdf content', 'application/pdf')}
                response = requests.post(
                    f'{API_BASE_URL}/api/v1/documents/upload',
                    files=files,
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5
                )
                
                if role == 'public':
                    if response.status_code == 403:
                        print(f'   ✅ Document upload correctly denied for {role} (status 403)')
                    else:
                        print(f'   ⚠️  Document upload should be denied for {role} (status {response.status_code})')
                else:
                    if response.status_code not in [401, 403]:
                        print(f'   ✅ Document upload accessible with {role} token (status {response.status_code})')
                    else:
                        print(f'   ⚠️  Document upload denied for {role} (status {response.status_code})')
            except Exception as e:
                print(f'   ⚠️  Error testing document upload with {role} token: {e}')
        
        # Metrics (Admin only)
        for role in ['public', 'solicitor', 'admin']:
            token = test_tokens[role]['access_token']
            try:
                response = requests.get(
                    f'{API_BASE_URL}/api/v1/metrics',
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5
                )
                
                if role == 'admin':
                    if response.status_code not in [401, 403]:
                        print(f'   ✅ Metrics endpoint accessible with {role} token (status {response.status_code})')
                    else:
                        print(f'   ⚠️  Metrics endpoint denied for {role} (status {response.status_code})')
                else:
                    if response.status_code == 403:
                        print(f'   ✅ Metrics endpoint correctly denied for {role} (status 403)')
                    else:
                        print(f'   ⚠️  Metrics endpoint should be denied for {role} (status {response.status_code})')
            except Exception as e:
                print(f'   ⚠️  Error testing metrics with {role} token: {e}')
        
        # Agentic chat - solicitor mode (Solicitor/Admin only)
        print('\n6. Testing Agentic Chat Mode Access Control...')
        for role in ['public', 'solicitor', 'admin']:
            token = test_tokens[role]['access_token']
            try:
                # Test solicitor mode
                response = requests.post(
                    f'{API_BASE_URL}/api/v1/agentic-chat',
                    json={'query': 'test', 'mode': 'solicitor'},
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5
                )
                
                if role == 'public':
                    if response.status_code == 403:
                        print(f'   ✅ Solicitor mode correctly denied for {role} (status 403)')
                    else:
                        print(f'   ⚠️  Solicitor mode should be denied for {role} (status {response.status_code})')
                else:
                    if response.status_code not in [401, 403]:
                        print(f'   ✅ Solicitor mode accessible with {role} token (status {response.status_code})')
                    else:
                        print(f'   ⚠️  Solicitor mode denied for {role} (status {response.status_code})')
            except Exception as e:
                print(f'   ⚠️  Error testing solicitor mode with {role} token: {e}')
        
        # Test public mode (all authenticated users)
        for role in ['public', 'solicitor', 'admin']:
            token = test_tokens[role]['access_token']
            try:
                response = requests.post(
                    f'{API_BASE_URL}/api/v1/agentic-chat',
                    json={'query': 'test', 'mode': 'public'},
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5
                )
                if response.status_code not in [401, 403]:
                    print(f'   ✅ Public mode accessible with {role} token (status {response.status_code})')
                else:
                    print(f'   ⚠️  Public mode denied for {role} (status {response.status_code})')
            except Exception as e:
                print(f'   ⚠️  Error testing public mode with {role} token: {e}')
    
    # Clean up test users
    print('\n7. Cleaning up test data...')
    db = SessionLocal()
    try:
        for role_name, user in test_users.items():
            db.query(RefreshToken).filter(RefreshToken.user_id == user.id).delete()
            db.delete(user)
        db.commit()
        print('   ✅ Test data cleaned up')
    finally:
        db.close()
    
    print('\n' + '='*60)
    print('✅ API ENDPOINT PROTECTION TEST COMPLETE')
    print('='*60)
    print('\nNote: If API server is not running, start it with:')
    print('  uvicorn app.api.main:app --reload')
    print('\nThen run this test again to verify HTTP endpoint protection.')
    
except ImportError as e:
    print(f'\n❌ Import error: {e}')
    print('Please install dependencies: pip install -r requirements.txt')
    sys.exit(1)
except Exception as e:
    print(f'\n❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

