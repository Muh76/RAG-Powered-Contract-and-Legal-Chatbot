# OAuth Login Verification (Google)

This document describes the OAuth (Google) login flow, token refresh, and protected-route behaviour, and provides a verification checklist. **No new auth logic was implemented**; this is validation and documentation only.

---

## 1. Prerequisites

### 1.1 Google Cloud Console

- Create (or use) a project in [Google Cloud Console](https://console.cloud.google.com/).
- Enable **Google+ API** or **Google Identity** (APIs & Services → Library → “Google+ API” or “People API”).
- **Credentials** → Create OAuth 2.0 Client ID (Application type: **Web application**).
- Under **Authorized redirect URIs**, add exactly:
  - `http://localhost:8501`
- Copy **Client ID** and **Client secret**.

### 1.2 Environment

In `.env` (or environment):

```bash
OAUTH_GOOGLE_CLIENT_ID=<your_client_id>
OAUTH_GOOGLE_CLIENT_SECRET=<your_client_secret>
JWT_SECRET_KEY=<strong_secret>
DATABASE_URL=postgresql://...
```

**Note:** The frontend sends `redirect_uri=http://localhost:8501` (no path) when requesting the auth URL and when exchanging the code. The redirect URI registered in Google **must match exactly** (`http://localhost:8501`). The config key `OAUTH_REDIRECT_URI` (e.g. `http://localhost:8501/auth/callback`) is not used by the backend for building the auth URL; the backend uses the `redirect_uri` query parameter from the client.

### 1.3 Services

- Backend: `uvicorn app.api.main:app --reload --port 8000`
- Frontend: `streamlit run frontend/app.py --server.port 8501`
- PostgreSQL running; migrations applied (`alembic upgrade head`).

---

## 2. Login Flow (Google)

### 2.1 Intended sequence

1. User opens `http://localhost:8501` → unauthenticated → login page (tabs: Login / Register).
2. User clicks **“🔵 Google”** (or equivalent Google OAuth button).
3. Frontend calls `GET /api/v1/auth/oauth/google/authorize?redirect_uri=http://localhost:8501`.
4. Backend (`AuthService.get_oauth_authorization_url`) builds Google’s auth URL (client_id, redirect_uri, scope, state) and returns `{ "authorization_url": "...", "state": "..." }`.
5. Frontend shows a link to that URL; user clicks and is sent to Google.
6. User signs in/consents; Google redirects to `http://localhost:8501?code=...&state=...`.
7. Streamlit loads with `code` (and optionally `state`) in `st.query_params`. Frontend checks `if "code" in query_params`, then calls `handle_oauth_callback(code, provider=query_params.get("provider") or session_state.oauth_provider)`. For Google, provider is typically taken from session (set when user clicked Google) or defaults to `"google"` in `handle_oauth_callback`.
8. Frontend calls `POST /api/v1/auth/oauth/google/callback` with body `{ "provider": "google", "code": "<code>", "redirect_uri": "http://localhost:8501" }`.
9. Backend (`AuthService.oauth_login`) exchanges `code` for Google tokens, fetches user info, then either links/creates user and OAuth account, and returns **our** JWT pair: `{ "access_token", "refresh_token", "token_type", "expires_in" }`.
10. Frontend stores tokens in session state, calls `GET /api/v1/auth/me` with `Authorization: Bearer <access_token>`, stores user, clears query params, reruns → user sees the main app (Chat/Documents/etc.).

### 2.2 Code references

| Step | Component | Location |
|------|-----------|----------|
| Auth URL | Backend | `app/auth/service.py` → `get_oauth_authorization_url`; `app/auth/oauth.py` → `GoogleOAuthProvider.get_authorization_url` |
| Auth URL | API | `GET /api/v1/auth/oauth/{provider}/authorize` → `app/api/routes/auth.py` |
| Initiate OAuth | Frontend | `frontend/auth_ui.py` → `_initiate_oauth("google")` (GET authorize, then show link) |
| Callback handler | Frontend | `frontend/app.py` → `run()` (detect `code` in query_params); `frontend/auth_ui.py` → `handle_oauth_callback` |
| Callback API | Backend | `POST /api/v1/auth/oauth/{provider}/callback` → `app/api/routes/auth.py` → `AuthService.oauth_login` |
| Token creation | Backend | `AuthService.oauth_login` → `AuthService.create_tokens` (same as email/password) |

### 2.3 What to verify manually

- [ ] Clicking “Google” shows a link that goes to `accounts.google.com`.
- [ ] After signing in with Google, browser returns to `http://localhost:8501?code=...&state=...`.
- [ ] No error on the callback; user is logged in and sees the main app (sidebar with profile).
- [ ] `GET /api/v1/auth/me` with the returned access token returns the user (email matches Google account).

---

## 3. Token Refresh

### 3.1 Behaviour

- After login (email or OAuth), the frontend holds **our** JWT `access_token` and `refresh_token` in session state.
- Access token expiry is configured by `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (e.g. 30). Refresh tokens are stored in DB and valid for `JWT_REFRESH_TOKEN_EXPIRE_DAYS` (e.g. 7).
- Frontend considers the access token expired when `datetime.now() >= token_expires_at - 60 seconds` (60 s buffer). Before protected API calls, `ensure_authenticated()` checks that; if expired, it calls `refresh_access_token()`.
- `refresh_access_token()` does `POST /api/v1/auth/refresh` with body `{ "refresh_token": "<stored_refresh_token>" }`. Backend verifies the refresh token (signature + DB, not revoked), then issues a new access token (and reuses the same refresh token). Frontend stores the new access token and new expiry.

### 3.2 Code references

| Step | Component | Location |
|------|-----------|----------|
| Expiry check | Frontend | `frontend/auth_ui.py` → `is_token_expired()`, `ensure_authenticated()` |
| Refresh request | Frontend | `frontend/auth_ui.py` → `refresh_access_token()` |
| Refresh API | Backend | `POST /api/v1/auth/refresh` → `app/api/routes/auth.py` → `AuthService.refresh_access_token` |
| Refresh logic | Backend | `app/auth/service.py` → `refresh_access_token` (verify JWT + DB, create new access token) |

### 3.3 What to verify manually

- [ ] Log in with Google; use the app for a while or wait until access token is expired (or temporarily lower `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`).
- [ ] Perform an action that triggers an API call (e.g. send a chat message, open Documents).
- [ ] No “Session expired” or 401; request succeeds. (Confirms refresh was used.)
- [ ] Optionally: call `POST /api/v1/auth/refresh` with the stored refresh_token; response contains a new `access_token` and `expires_in`.

---

## 4. Protected Routes

### 4.1 Backend

- Protected routes use `Depends(get_current_user)` or role-specific deps (`require_admin`, `require_solicitor_or_admin`). `get_current_user`:
  - Reads `Authorization: Bearer <token>`.
  - Verifies the JWT as access token (`verify_token(..., token_type="access")`).
  - Loads user from DB by `token_data.user_id`; checks `is_active`.
  - Returns 401 if token missing/invalid or user not found/inactive; 403 if role check fails.
- OAuth vs email/password makes no difference: both receive the same JWT after login; protection is purely via JWT and DB user.

### 4.2 Frontend

- Main app (Chat, Documents, Admin, Settings) is shown only after `ensure_authenticated()` returns True (token present and not expired, or refreshed).
- If not authenticated, the login page is shown. No Bearer token is sent until the user has logged in (email or OAuth) and tokens are stored.

### 4.3 What to verify manually

- [ ] Without logging in, open `http://localhost:8501` → login page only; no Chat/Documents content.
- [ ] After Google login, Chat/Documents/Admin (if admin) work; API requests include `Authorization: Bearer <access_token>`.
- [ ] Call a protected API (e.g. `GET /api/v1/auth/me` or `POST /api/v1/chat`) without a token or with an invalid token → 401 (or 403 for Bearer with invalid token).
- [ ] As a public user, call an admin-only endpoint (e.g. `GET /api/v1/auth/users`) → 403.

---

## 5. Findings (Code-Level)

- **Redirect URI:** Frontend uses a fixed `redirect_uri=http://localhost:8501` (no path). Google Cloud Console must list exactly `http://localhost:8501` as an authorized redirect URI. The server config `OAUTH_REDIRECT_URI` (e.g. `http://localhost:8501/auth/callback`) is not used when building the auth URL.
- **State parameter:** Backend generates and returns `state` in the auth URL response. The callback endpoint does **not** validate `state`; the frontend does not send `state` in the callback request. So state is not used for CSRF protection on the callback. For production, consider validating state on the backend when exchanging the code.
- **Provider on callback:** When Google redirects to `http://localhost:8501?code=...&state=...`, the URL does not include `provider`. The frontend relies on `session_state.oauth_provider` (set when the user clicked “Google”) or defaults to `"google"` in `handle_oauth_callback`. If the user bookmarks the callback URL or the session is lost, the default is still Google; for multi-provider setups, consider including `provider` in the redirect (e.g. as a query param) or in state.
- **Token refresh for OAuth users:** Uses the same JWT refresh flow as email/password users. Google’s own refresh token (stored in `oauth_accounts`) is separate and is not used for our API access; only our JWT refresh token is used.

---

## 6. Verification Checklist Summary

| Area | Check |
|------|--------|
| **Login flow** | Google button → Google consent → redirect to app with `code` → logged in; profile in sidebar. |
| **Token refresh** | After access token expiry, next API call succeeds without re-login (refresh used). |
| **Protected routes** | Unauthenticated: login page only. Authenticated: Chat/Documents/Admin (by role) work with Bearer token. |
| **API security** | Missing/invalid token → 401; wrong role → 403. |

---

## 7. Optional: Quick API Checks (no browser)

- **Auth URL (Google):**  
  `curl -s "http://localhost:8000/api/v1/auth/oauth/google/authorize?redirect_uri=http://localhost:8501"`  
  → JSON with `authorization_url` (contains `accounts.google.com`) and `state`.

- **Refresh (after login):**  
  `curl -s -X POST http://localhost:8000/api/v1/auth/refresh -H "Content-Type: application/json" -d '{"refresh_token":"<your_refresh_token>"}'`  
  → JSON with `access_token`, `refresh_token`, `expires_in`.

- **Protected route (with token):**  
  `curl -s -H "Authorization: Bearer <access_token>" http://localhost:8000/api/v1/auth/me`  
  → User object. Without token or with invalid token → 401.

---

*Document generated for OAuth (Google) validation. No new auth logic was added.*
