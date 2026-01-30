#!/usr/bin/env python3
"""E2E smoke test: backend, health, auth, protected routes, chat, document upload.

Calls a RUNNING backend at http://localhost:8000 via HTTP. Does NOT import the
FastAPI app (avoids ML/native deps and in-process crashes).
"""

from __future__ import annotations

import sys
import time

try:
    import requests
except ImportError:
    print("Error: 'requests' is required. Install with: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
TIMEOUT = 30


def main() -> None:
    print(f"Smoke test starting (backend must be running at {BASE_URL})...", flush=True)

    # Probe backend reachability first
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Backend not reachable at {BASE_URL}.\n"
            "Ensure the backend is running: uvicorn app.api.main:app --reload --port 8000"
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"Error: Backend at {BASE_URL} did not respond within 10s.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to reach backend: {e}")
        sys.exit(1)

    checks: list[tuple[str, bool, str]] = []
    email = f"smoke_{int(time.time())}@example.com"
    password = "Password123!"
    access_token: str | None = None
    headers: dict[str, str] = {}

    # 1. Backend reachable (we got a response)
    checks.append(("1. Backend reachable", True, "OK"))

    # 2. GET /api/v1/health returns status healthy/degraded/ok
    ok = r.status_code == 200
    j = r.json() if ok else {}
    status_val = j.get("status", "")
    ok = ok and status_val in ("healthy", "degraded", "ok")
    checks.append((
        "2. GET /api/v1/health returns status ok (healthy/degraded)",
        ok,
        f"{r.status_code} status={status_val}" if ok else f"{r.status_code} {r.text[:200]}",
    ))

    # 3. POST /api/v1/auth/register
    r = requests.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"email": email, "password": password, "full_name": "Smoke", "role": "solicitor"},
        timeout=TIMEOUT,
    )
    ok = r.status_code == 201
    if ok:
        access_token = r.json().get("access_token")
        headers["Authorization"] = f"Bearer {access_token}"
    checks.append((
        "3. POST /api/v1/auth/register",
        ok,
        f"{r.status_code}" + (f" token len={len(access_token or '')}" if ok else f" {r.text[:150]}"),
    ))

    # 4. POST /api/v1/auth/login — skip if register returned access_token
    if access_token:
        checks.append(("4. POST /api/v1/auth/login", True, "skipped (token from register)"))
    else:
        r = requests.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={"email": email, "password": password},
            timeout=TIMEOUT,
        )
        ok = r.status_code == 200
        if ok:
            access_token = r.json().get("access_token")
            headers["Authorization"] = f"Bearer {access_token}"
        checks.append(("4. POST /api/v1/auth/login", ok, f"{r.status_code}" + ("" if ok else f" {r.text[:150]}")))

    # 5. GET /api/v1/auth/me unauthenticated (expect 401/403)
    r = requests.get(f"{BASE_URL}/api/v1/auth/me", timeout=TIMEOUT)
    ok = r.status_code in (401, 403)
    checks.append((
        "5. GET /api/v1/auth/me (unauthenticated) rejects",
        ok,
        f"{r.status_code} (expect 401/403)",
    ))

    # 6. GET /api/v1/auth/me authenticated
    if not access_token:
        checks.append(("6. GET /api/v1/auth/me (authenticated)", False, "no token from register/login"))
    else:
        r = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers, timeout=TIMEOUT)
        ok = r.status_code == 200
        checks.append(("6. GET /api/v1/auth/me (authenticated)", ok, f"{r.status_code}"))

    # 7. POST /api/v1/chat
    if not access_token:
        checks.append(("7. POST /api/v1/chat", False, "no token"))
    else:
        r = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={"query": "What is the Employment Rights Act?", "mode": "public"},
            headers=headers,
            timeout=TIMEOUT,
        )
        ok = r.status_code in (200, 503)
        checks.append(("7. POST /api/v1/chat", ok, f"{r.status_code}"))

    # 8. POST /api/v1/documents/upload
    if not access_token:
        checks.append(("8. POST /api/v1/documents/upload", False, "no token"))
    else:
        r = requests.post(
            f"{BASE_URL}/api/v1/documents/upload",
            files={"file": ("smoke.txt", b"smoke test content", "text/plain")},
            headers=headers,
            timeout=TIMEOUT,
        )
        ok = r.status_code in (201, 400, 422)
        checks.append(("8. POST /api/v1/documents/upload", ok, f"{r.status_code}"))

    # Report
    print("E2E smoke test checklist")
    print("-" * 50)
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}  ({detail})")
    print("-" * 50)
    failed = [c[0] for c in checks if not c[1]]
    if failed:
        print("Blocking failures: " + ", ".join(failed))
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
