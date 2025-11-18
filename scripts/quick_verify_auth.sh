#!/bin/bash
# Quick Verification Script for Authentication and RBAC
# This script runs all verification tests

set -e

echo "=========================================="
echo "Authentication & RBAC Quick Verification"
echo "=========================================="

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 &> /dev/null; then
    echo "⚠️  PostgreSQL is not running"
    echo "Please start PostgreSQL: brew services start postgresql@14"
    exit 1
fi

echo "✅ PostgreSQL is running"

# Set environment variables
export DATABASE_URL="${DATABASE_URL:-postgresql://javadbeni@localhost:5432/legal_chatbot}"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-test-secret-key-for-testing}"
export SECRET_KEY="${SECRET_KEY:-test-secret-key}"

echo ""
echo "Running Route Protection Test..."
echo "----------------------------------------"
python scripts/test_route_protection.py

echo ""
echo "=========================================="
echo "✅ Verification Complete!"
echo "=========================================="
echo ""
echo "To test HTTP endpoints, start the API server:"
echo "  uvicorn app.api.main:app --reload"
echo ""
echo "Then run:"
echo "  python scripts/test_api_endpoints.py"
echo ""

