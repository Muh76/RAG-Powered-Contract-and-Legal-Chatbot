#!/usr/bin/env python3
"""E2E smoke test: backend, health, auth, protected routes, chat, document upload."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("LOG_FORMAT", "standard")

def main() -> None:
    print("Smoke test starting...", flush=True)
    from fastapi.testclient import TestClient
    from app.api.main import app

    client = TestClient(app)
    checks: list[tuple[str, bool, str]] = []
    email = f"smoke_{int(time.time())}@example.com"
    password = "Password123!"
    access_token: str | None = None

    # 1. Backend starts cleanly (we got here = app imported and TestClient created)
    checks.append(("1. Backend starts cleanly", True, "OK"))

    # 2. /api/v1/health returns status healthy (API uses 'healthy'|'degraded', not 'ok')
    r = client.get("/api/v1/health")
    ok = r.status_code == 200
    j = r.json() if ok else {}
    status_val = j.get("status", "")
    ok = ok and status_val in ("healthy", "degraded", "ok")
    checks.append((
        "2. /api/v1/health returns status ok (healthy/degraded)",
        ok,
        f"{r.status_code} status={status_val}" if ok else f"{r.status_code} {r.text[:200]}",
    ))

    # 3. Register
    r = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "full_name": "Smoke", "role": "solicitor"},
    )
    ok = r.status_code == 201
    if ok:
        access_token = r.json().get("access_token")
    checks.append(("3. Register works", ok, f"{r.status_code}" + (f" token len={len(access_token or '')}" if ok else f" {r.text[:150]}")))

    # 4. Login
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    ok = r.status_code == 200
    if ok and not access_token:
        access_token = r.json().get("access_token")
    checks.append(("4. Login works", ok, f"{r.status_code}" + ("" if ok else f" {r.text[:150]}")))

    # 5. Auth-protected route rejects unauthenticated
    r = client.get("/api/v1/auth/me")
    ok = r.status_code in (401, 403)
    checks.append((
        "5. Protected routes reject unauthenticated",
        ok,
        f"{r.status_code} (expect 401/403)",
    ))

    # 6. Auth-protected route allows authenticated
    if not access_token:
        checks.append(("6. Protected routes allow authenticated", False, "no token from register/login"))
    else:
        r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
        ok = r.status_code == 200
        checks.append(("6. Protected routes allow authenticated", ok, f"{r.status_code}"))

    # 7. Chat endpoint responds (200 or 503 e.g. RAG down)
    if not access_token:
        checks.append(("7. Chat endpoint responds", False, "no token"))
    else:
        r = client.post(
            "/api/v1/chat",
            json={"query": "What is the Employment Rights Act?", "mode": "public"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        ok = r.status_code in (200, 503)
        checks.append(("7. Chat endpoint responds", ok, f"{r.status_code}"))

    # 8. Document upload endpoint responds (201 or 4xx; solicitor role)
    if not access_token:
        checks.append(("8. Document upload endpoint responds", False, "no token"))
    else:
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("smoke.txt", b"smoke test content", "text/plain")},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        ok = r.status_code in (201, 400, 422)
        checks.append(("8. Document upload endpoint responds", ok, f"{r.status_code}"))

    # Report
    print("E2E smoke test checklist")
    print("-" * 50)
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}  ({detail})")
    print("-" * 50)
    failed = [c[0] for c in checks if not c[1]]
    if failed:
        print("Blocking failures:", ", ".join(failed))
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
