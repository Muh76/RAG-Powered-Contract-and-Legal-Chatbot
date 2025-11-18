"""
Test protected routes with authentication
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.main import app
from app.auth.service import AuthService
from app.auth.schemas import UserCreate, UserRole
from app.core.database import SessionLocal


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def public_user(db: Session):
    """Create a public user for testing"""
    user_data = UserCreate(
        email="public@test.com",
        password="testpass123",
        full_name="Public User",
        role=UserRole.PUBLIC
    )
    user = AuthService.create_user(db, user_data)
    tokens = AuthService.create_tokens(user, db)
    return user, tokens


@pytest.fixture
def solicitor_user(db: Session):
    """Create a solicitor user for testing"""
    user_data = UserCreate(
        email="solicitor@test.com",
        password="testpass123",
        full_name="Solicitor User",
        role=UserRole.SOLICITOR
    )
    user = AuthService.create_user(db, user_data)
    tokens = AuthService.create_tokens(user, db)
    return user, tokens


@pytest.fixture
def admin_user(db: Session):
    """Create an admin user for testing"""
    user_data = UserCreate(
        email="admin@test.com",
        password="testpass123",
        full_name="Admin User",
        role=UserRole.ADMIN
    )
    user = AuthService.create_user(db, user_data)
    tokens = AuthService.create_tokens(user, db)
    return user, tokens


def test_health_endpoint_public(client):
    """Health endpoint should be public (no auth required)"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_chat_endpoint_requires_auth(client):
    """Chat endpoint should require authentication"""
    response = client.post(
        "/api/v1/chat",
        json={"query": "What is a contract?"}
    )
    assert response.status_code == 403  # Forbidden without auth


def test_chat_endpoint_with_auth(client, public_user):
    """Chat endpoint should work with valid token"""
    user, tokens = public_user
    response = client.post(
        "/api/v1/chat",
        json={"query": "What is a contract?"},
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    # Should either succeed (if services initialized) or fail with service error, not auth error
    assert response.status_code != 403
    assert response.status_code != 401


def test_search_endpoint_requires_auth(client):
    """Search endpoint should require authentication"""
    response = client.post(
        "/api/v1/search/hybrid",
        json={"query": "contract law", "top_k": 5}
    )
    assert response.status_code == 403


def test_search_endpoint_with_auth(client, public_user):
    """Search endpoint should work with valid token"""
    user, tokens = public_user
    response = client.post(
        "/api/v1/search/hybrid",
        json={"query": "contract law", "top_k": 5},
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code != 403
    assert response.status_code != 401


def test_document_upload_requires_solicitor_or_admin(client, public_user):
    """Document upload should require Solicitor or Admin role"""
    user, tokens = public_user
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code == 403  # Public user should be denied


def test_document_upload_with_solicitor(client, solicitor_user):
    """Document upload should work with Solicitor role"""
    user, tokens = solicitor_user
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code != 403  # Solicitor should be allowed


def test_agentic_chat_solicitor_mode_requires_solicitor(client, public_user):
    """Agentic chat in solicitor mode should require Solicitor or Admin role"""
    user, tokens = public_user
    response = client.post(
        "/api/v1/agentic-chat",
        json={
            "query": "What is a contract?",
            "mode": "solicitor"
        },
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code == 403  # Public user should be denied solicitor mode


def test_agentic_chat_public_mode_allowed(client, public_user):
    """Agentic chat in public mode should work for all authenticated users"""
    user, tokens = public_user
    response = client.post(
        "/api/v1/agentic-chat",
        json={
            "query": "What is a contract?",
            "mode": "public"
        },
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code != 403
    assert response.status_code != 401


def test_metrics_endpoint_requires_admin(client, public_user):
    """Metrics endpoint should require Admin role"""
    user, tokens = public_user
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code == 403  # Public user should be denied


def test_metrics_endpoint_with_admin(client, admin_user):
    """Metrics endpoint should work with Admin role"""
    user, tokens = admin_user
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert response.status_code != 403  # Admin should be allowed

