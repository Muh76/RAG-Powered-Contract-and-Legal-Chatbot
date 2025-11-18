# Database Setup Guide - Legal Chatbot

## 🎯 Overview

This guide will help you set up PostgreSQL database and run migrations for the authentication system.

## 📋 Prerequisites

1. PostgreSQL installed and running
2. Python dependencies installed (`pip install -r requirements.txt`)
3. Database connection details

## 🚀 Setup Options

### Option 1: Docker Compose (Recommended)

**If you have Docker installed:**

```bash
# Start PostgreSQL container
docker compose up -d postgres

# Wait for PostgreSQL to be ready (5-10 seconds)
sleep 5

# Verify it's running
docker compose ps postgres
```

**Database Details:**
- Host: `localhost`
- Port: `5432`
- Database: `legal_chatbot`
- User: `postgres`
- Password: `password`

### Option 2: Local PostgreSQL Installation

**On macOS (Homebrew):**

```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Verify it's running
pg_isready -h localhost -p 5432
```

**On Linux:**

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu/Debian
# or
sudo yum install postgresql-server postgresql-contrib  # CentOS/RHEL

# Start PostgreSQL service
sudo service postgresql start
# or
sudo systemctl start postgresql
```

**On Windows:**

1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Install with default settings
3. PostgreSQL service should start automatically

## 📝 Configuration

### 1. Create `.env` File

Create or update `.env` file in project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/legal_chatbot

# Security Configuration
JWT_SECRET_KEY=your-secret-key-change-in-production
SECRET_KEY=your-secret-key-change-in-production
```

**Generate secure keys:**

```bash
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
```

### 2. Create Database

**If using Docker Compose:**
- Database is created automatically when container starts

**If using local PostgreSQL:**

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE legal_chatbot;

# Exit psql
\q
```

Or use the setup script:

```bash
chmod +x scripts/setup_database.sh
./scripts/setup_database.sh
```

## 🔄 Running Migrations

### Step 1: Generate Migration (if needed)

```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key"
export SECRET_KEY="test-secret-key"

python -m alembic revision --autogenerate -m "create_auth_tables"
```

### Step 2: Apply Migrations

```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key"
export SECRET_KEY="test-secret-key"

python -m alembic upgrade head
```

### Step 3: Verify Migration

```bash
python -m alembic current
```

Expected output:
```
rev_id (head)
```

## ✅ Verification

### Check Tables

```bash
psql -h localhost -U postgres -d legal_chatbot -c "\dt"
```

Expected output:
```
                    List of relations
 Schema |       Name        | Type  |  Owner   
--------+-------------------+-------+----------
 public | alembic_version   | table | postgres
 public | oauth_accounts    | table | postgres
 public | refresh_tokens    | table | postgres
 public | users             | table | postgres
```

### Test Authentication

```bash
python scripts/test_auth_database.py
```

## 🐛 Troubleshooting

### Error: "Connection refused"

**Problem**: PostgreSQL is not running

**Solution**:
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Start PostgreSQL
# Docker:
docker compose up -d postgres

# Homebrew (macOS):
brew services start postgresql@15

# System service (Linux):
sudo service postgresql start
```

### Error: "Database does not exist"

**Problem**: Database `legal_chatbot` not created

**Solution**:
```bash
psql -h localhost -U postgres -d postgres -c "CREATE DATABASE legal_chatbot;"
```

### Error: "ModuleNotFoundError: No module named 'psycopg2'"

**Problem**: PostgreSQL adapter not installed

**Solution**:
```bash
pip install psycopg2-binary
```

### Error: "Authentication failed"

**Problem**: Wrong database credentials

**Solution**:
1. Check `.env` file has correct `DATABASE_URL`
2. Verify PostgreSQL user/password
3. For Docker: Default is `postgres:password`

## 📊 Database Schema

After migrations, you'll have these tables:

### `users` Table
- User accounts with roles (Admin, Solicitor, Public)
- Email, username, password, profile info

### `oauth_accounts` Table
- OAuth account linking (Google, GitHub, Microsoft)
- Links external accounts to users

### `refresh_tokens` Table
- Refresh token storage
- Token expiration and revocation

### `alembic_version` Table
- Migration tracking
- Managed by Alembic

## 🔍 Next Steps

After database setup:

1. **Verify Setup**
   ```bash
   python scripts/test_auth_database.py
   ```

2. **Protect Existing Routes**
   - Add authentication to chat/search endpoints
   - Implement role-based access control

3. **Test API Endpoints**
   ```bash
   # Start server
   uvicorn app.api.main:app --reload
   
   # Test registration
   curl -X POST http://localhost:8000/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"testpass123"}'
   ```

## 📝 Notes

- **Production**: Change default passwords and secret keys
- **Security**: Use strong `JWT_SECRET_KEY` and `SECRET_KEY`
- **Backup**: Regular database backups recommended
- **Migrations**: Never edit existing migration files, create new ones

## ✨ Summary

✅ **Database setup steps:**
1. Start PostgreSQL (Docker or local)
2. Create `.env` file with database URL
3. Run migrations: `alembic upgrade head`
4. Verify tables created
5. Test authentication system

**Status**: Ready for testing! 🎉

