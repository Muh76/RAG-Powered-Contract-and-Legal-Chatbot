#!/bin/bash
# Database Setup Script for Legal Chatbot
# This script sets up PostgreSQL database for authentication system

set -e

echo "=========================================="
echo "Legal Chatbot - Database Setup"
echo "=========================================="

# Check if PostgreSQL is running
if command -v pg_isready &> /dev/null; then
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        echo "✅ PostgreSQL is running"
    else
        echo "⚠️  PostgreSQL is not running on localhost:5432"
        echo ""
        echo "Please start PostgreSQL using one of these methods:"
        echo ""
        echo "Option 1: Docker Compose"
        echo "  docker compose up -d postgres"
        echo ""
        echo "Option 2: Homebrew (macOS)"
        echo "  brew services start postgresql@15"
        echo "  # or"
        echo "  brew services start postgresql"
        echo ""
        echo "Option 3: System service"
        echo "  sudo service postgresql start"
        echo "  # or"
        echo "  sudo systemctl start postgresql"
        echo ""
        exit 1
    fi
else
    echo "⚠️  PostgreSQL client (pg_isready) not found"
    echo "Please install PostgreSQL first"
    exit 1
fi

# Create database
echo ""
echo "Creating database..."
psql -h localhost -U postgres -d postgres -c "CREATE DATABASE legal_chatbot;" 2>/dev/null || echo "Database already exists"

# Verify database connection
echo "Verifying database connection..."
psql -h localhost -U postgres -d legal_chatbot -c "SELECT version();" > /dev/null
echo "✅ Database connection successful"

# Run migrations
echo ""
echo "Running database migrations..."
export DATABASE_URL="postgresql://postgres:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key-change-in-production"
export SECRET_KEY="test-secret-key-change-in-production"

python -m alembic upgrade head

echo ""
echo "=========================================="
echo "✅ Database setup complete!"
echo "=========================================="
echo ""
echo "Database: legal_chatbot"
echo "Tables created: users, oauth_accounts, refresh_tokens"
echo ""
echo "Next steps:"
echo "  1. Update .env file with DATABASE_URL"
echo "  2. Set JWT_SECRET_KEY and SECRET_KEY in .env"
echo "  3. Test authentication endpoints"

