# Route Protection Verification Guide

## 🎯 Overview

This guide helps you verify that authentication and role-based access control (RBAC) are working correctly in the Legal Chatbot API.

## ✅ Verification Steps

### Step 1: Run Route Protection Test Script

This script verifies the authentication implementation without needing the API server:

```bash
# Set environment variables
export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key-for-testing"
export SECRET_KEY="test-secret-key"

# Run test script
python scripts/test_route_protection.py
```

**What it checks:**
- ✅ Authentication dependencies are correctly imported
- ✅ All route files have authentication protection
- ✅ Health endpoint remains public (no auth required)
- ✅ Role-based access control is enforced
- ✅ Token creation and verification works
- ✅ Token refresh works
- ✅ Different user roles have appropriate access

### Step 2: Test API Endpoints with HTTP Requests

This script tests the actual HTTP endpoints (requires API server running):

```bash
# Start API server (in a separate terminal)
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run endpoint tests
export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key-for-testing"
export SECRET_KEY="test-secret-key"
export API_BASE_URL="http://localhost:8000"

python scripts/test_api_endpoints.py
```

**What it checks:**
- ✅ Public endpoints work without authentication
- ✅ Protected endpoints require authentication
- ✅ RBAC endpoints enforce role-based access
- ✅ Different roles have appropriate permissions

### Step 3: Manual Testing with curl

#### 3.1 Test Public Endpoint (No Auth Required)

```bash
# Health check should work without authentication
curl -X GET http://localhost:8000/api/v1/health
```

**Expected**: Status 200

#### 3.2 Test Protected Endpoint (Auth Required)

```bash
# Chat endpoint should require authentication
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

**Expected**: Status 401 or 403 (Unauthorized)

#### 3.3 Register a User and Get Token

```bash
# Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "Test User",
    "role": "public"
  }'
```

**Response**: Returns `access_token` and `refresh_token`

#### 3.4 Test Protected Endpoint with Token

```bash
# Replace YOUR_ACCESS_TOKEN with the token from registration
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{"query": "What is a contract?"}'
```

**Expected**: Status 200 (if services initialized) or 500 (if services not initialized, but auth works)

#### 3.5 Test RBAC - Document Upload (Solicitor/Admin Only)

```bash
# Try with Public user token (should fail)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer PUBLIC_USER_TOKEN" \
  -F "file=@test.pdf"
```

**Expected**: Status 403 (Forbidden - Public users denied)

```bash
# Try with Solicitor user token (should work)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer SOLICITOR_USER_TOKEN" \
  -F "file=@test.pdf"
```

**Expected**: Status 200 (Solicitor users allowed)

#### 3.6 Test RBAC - Metrics (Admin Only)

```bash
# Try with Public user token (should fail)
curl -X GET http://localhost:8000/api/v1/metrics \
  -H "Authorization: Bearer PUBLIC_USER_TOKEN"
```

**Expected**: Status 403 (Forbidden - Public users denied)

```bash
# Try with Admin user token (should work)
curl -X GET http://localhost:8000/api/v1/metrics \
  -H "Authorization: Bearer ADMIN_USER_TOKEN"
```

**Expected**: Status 200 (Admin users allowed)

#### 3.7 Test Agentic Chat Mode Access

```bash
# Try solicitor mode with Public user (should fail)
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer PUBLIC_USER_TOKEN" \
  -d '{"query": "test", "mode": "solicitor"}'
```

**Expected**: Status 403 (Forbidden - Public users cannot use solicitor mode)

```bash
# Try solicitor mode with Solicitor user (should work)
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer SOLICITOR_USER_TOKEN" \
  -d '{"query": "test", "mode": "solicitor"}'
```

**Expected**: Status 200 (Solicitor users can use solicitor mode)

```bash
# Try public mode with Public user (should work)
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer PUBLIC_USER_TOKEN" \
  -d '{"query": "test", "mode": "public"}'
```

**Expected**: Status 200 (All authenticated users can use public mode)

### Step 4: Test with Swagger UI

1. Start the API server:
   ```bash
   uvicorn app.api.main:app --reload
   ```

2. Open Swagger UI:
   ```
   http://localhost:8000/docs
   ```

3. Test authentication:
   - Click "Authorize" button (🔓 icon)
   - Enter: `Bearer YOUR_ACCESS_TOKEN`
   - Click "Authorize"

4. Test protected endpoints:
   - Try `/api/v1/chat` - Should work with token
   - Try `/api/v1/documents/upload` - Should work with Solicitor/Admin token
   - Try `/api/v1/metrics` - Should work with Admin token only

## 📊 Expected Results

### Public Endpoints (No Auth Required)
- ✅ `/api/v1/health` - Status 200
- ✅ `/docs` - Swagger UI accessible
- ✅ `/` - Root endpoint accessible

### Protected Endpoints (Auth Required)
- ❌ Without token: Status 401/403
- ✅ With valid token: Status 200 (or service-specific status)

### RBAC Endpoints

#### Document Endpoints (Solicitor/Admin Only)
- ❌ Public user: Status 403
- ✅ Solicitor user: Status 200
- ✅ Admin user: Status 200

#### Metrics Endpoints (Admin Only)
- ❌ Public user: Status 403
- ❌ Solicitor user: Status 403
- ✅ Admin user: Status 200

#### Agentic Chat (Mode-Based)
- ✅ All authenticated users: Public mode allowed
- ❌ Public user: Solicitor mode denied (Status 403)
- ✅ Solicitor user: Solicitor mode allowed
- ✅ Admin user: Solicitor mode allowed

## 🔍 Verification Checklist

- [ ] Route protection test script passes
- [ ] API endpoint test script passes
- [ ] Public endpoints accessible without auth
- [ ] Protected endpoints require authentication
- [ ] Document upload requires Solicitor/Admin role
- [ ] Metrics endpoints require Admin role
- [ ] Agentic chat solicitor mode requires Solicitor/Admin role
- [ ] Token refresh works correctly
- [ ] Invalid tokens are rejected
- [ ] Expired tokens are rejected
- [ ] Inactive users are denied access

## 🐛 Troubleshooting

### Error: "Connection refused" when testing API endpoints

**Problem**: API server is not running

**Solution**:
```bash
uvicorn app.api.main:app --reload
```

### Error: "ModuleNotFoundError: psycopg2"

**Problem**: PostgreSQL adapter not installed

**Solution**:
```bash
pip install psycopg2-binary
```

### Error: "Database connection failed"

**Problem**: PostgreSQL not running

**Solution**:
```bash
# Start PostgreSQL
brew services start postgresql@14

# Or with Docker
docker compose up -d postgres
```

### Error: "401 Unauthorized" with valid token

**Problem**: Token expired or invalid

**Solution**:
- Register/login again to get a new token
- Check JWT_SECRET_KEY matches between token creation and verification

### Error: "403 Forbidden" with valid token

**Problem**: User role doesn't have required permissions

**Solution**:
- Check user role: Use `/api/v1/auth/me` endpoint
- Ensure user has correct role for the endpoint

## ✨ Summary

After running all verification tests, you should have:
- ✅ All route protection tests passing
- ✅ All API endpoint tests passing
- ✅ Clear understanding of which endpoints require authentication
- ✅ Clear understanding of role-based access restrictions
- ✅ Confidence that authentication and RBAC are working correctly

---

**Quick Test Command**:
```bash
# One-command test (requires PostgreSQL running)
export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot" && \
export JWT_SECRET_KEY="test-secret-key" && \
export SECRET_KEY="test-secret-key" && \
python scripts/test_route_protection.py
```

