# Cloud Run: Why the container failed to start (404)

## What happened

Cloud Run reported: **"The user-provided container failed to start and listen on the port defined by PORT=8080 within the allocated timeout"** (HealthCheckContainerError). So the process never bound to port 8080, and all requests (including `/`) returned **404**.

## Root causes (any one can cause failure)

1. **Missing DEMO_MODE**  
   Without `DEMO_MODE=true`, the app calls `validate_required_config()` and requires `DATABASE_URL` and `JWT_SECRET_KEY`. If those env vars are not set in Cloud Run, startup raises and the process exits before listening.

2. **DEMO_MODE not recognized**  
   If `DEMO_MODE` is set as the string `"true"` and the app only checked the Pydantic `settings` object, it might not have been treated as true. The app now also checks `os.getenv("DEMO_MODE")` so `DEMO_MODE=true` is always respected.

3. **Strict embedding config**  
   `_validate_embedding_config()` runs at startup. In DEMO_MODE we now skip it so the container can start without `OPENAI_API_KEY`.

4. **RAG init (background)**  
   RAG loads FAISS from `data/`. If the image has a FAISS index but no `OPENAI_API_KEY`, the **background** init can raise. That no longer blocks the server from binding, because init runs in a thread after the server has started. If RAG fails, the app runs in a degraded state and logs the error.

## What was changed

- **Startup:** `DEMO_MODE` is detected via both `settings` and `os.getenv("DEMO_MODE")`.
- **Startup:** In DEMO_MODE, `_validate_embedding_config()` is skipped so startup does not require embedding/OpenAI config.
- **Startup:** Exceptions in lifespan are logged with full traceback before re-raising, so Cloud Run logs show the exact error if something else fails.
- **Deploy:** `scripts/deploy.sh` now sets `DEMO_MODE=true` in Cloud Run so the container can start without DB/JWT.

## What you should do

1. **Redeploy** with the updated code and deploy script (so `DEMO_MODE=true` is set). The service URL should then respond (e.g. `/` and `/health`).

2. **If it still fails,** open **Cloud Run → your service → Logs** and look for the line:  
   `Startup failed (check Cloud Run logs): ...`  
   The traceback right below it is the real cause.

3. **Optional:** Set `OPENAI_API_KEY` (and any other secrets) in Cloud Run → Edit & deploy → Variables & secrets so RAG and full features work.

---

## Finding logs when "0 results" appears

If Logs Explorer shows **0 results**:

1. **Set the time range** to when the deploy ran. The deploy that failed ended around **17:47 UTC** (18 Feb 2026). In the time picker choose "Last 1 hour" or set a custom range that includes that time (e.g. 17:00–18:00 UTC).

2. **Filter by Cloud Run revision** so you see only this service:
   - In the query box, use:
   ```
   resource.type="cloud_run_revision"
   resource.labels.service_name="legal-chatbot-api"
   ```
   Or: **Logs** → **Cloud Run** → select service **legal-chatbot-api** (that applies the filter for you).

3. **Look for the bootstrap line**  
   After redeploying, you should see at least:
   ```
   legal-chatbot-api: process starting
   ```
   If you see it, the container is starting and the crash is later (imports or lifespan). If you never see it, the process is not running (e.g. wrong image or entrypoint).
