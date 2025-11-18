# Database Setup - Quick Start Guide

## ⚠️ **PostgreSQL Required**

The authentication system requires PostgreSQL to be running. 

## 🚀 **Quick Setup (Choose One Method)**

### Method 1: Docker Compose (Easiest)

```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Wait for PostgreSQL to be ready (5-10 seconds)
sleep 5

# 3. Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="your-secret-key-here"
export SECRET_KEY="your-secret-key-here"

# 4. Run migrations
python -m alembic upgrade head

# 5. Test setup
python scripts/test_auth_database.py
```

### Method 2: Local PostgreSQL

```bash
# 1. Install PostgreSQL (if not installed)
# macOS:
brew install postgresql@15
brew services start postgresql@15

# Linux:
sudo apt-get install postgresql postgresql-contrib
sudo service postgresql start

# 2. Create database
psql -U postgres -c "CREATE DATABASE legal_chatbot;"

# 3. Set environment variables
export DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="your-secret-key-here"
export SECRET_KEY="your-secret-key-here"

# 4. Run migrations
python -m alembic upgrade head

# 5. Test setup
python scripts/test_auth_database.py
```

### Method 3: Using Setup Script

```bash
# Make script executable
chmod +x scripts/setup_database.sh

# Run setup (requires PostgreSQL running)
./scripts/setup_database.sh
```

## ✅ **What Gets Created**

After running migrations, these tables will be created:

- ✅ `users` - User accounts with roles
- ✅ `oauth_accounts` - OAuth account linking
- ✅ `refresh_tokens` - Refresh token storage
- ✅ `alembic_version` - Migration tracking

## 🔍 **Verify Setup**

```bash
# Check database status
python scripts/test_auth_database.py

# Or check manually
psql -h localhost -U postgres -d legal_chatbot -c "\dt"
```

## 📝 **Troubleshooting**

**Problem**: "Connection refused"
- **Solution**: Start PostgreSQL (see methods above)

**Problem**: "Database does not exist"
- **Solution**: Create database: `psql -U postgres -c "CREATE DATABASE legal_chatbot;"`

**Problem**: "ModuleNotFoundError: psycopg2"
- **Solution**: `pip install psycopg2-binary`

## 📚 **Full Documentation**

See `docs/database_setup_guide.md` for detailed instructions.

---

**Quick Status Check**: Run `python scripts/test_auth_database.py` to verify everything is working.

