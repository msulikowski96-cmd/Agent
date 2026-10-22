"""Serwer HTTP: panel mobilny + API. Uruchom: python app.py  (zero zależności — tylko biblioteka standardowa)."""
import hashlib
import hmac
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import time
import traceback
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import IGNORED_DIRS, PROVIDERS, LLMError, Runner, get_api_key, list_models
from store import Store

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()
PORT = int(os.environ.get("PORT", "8080"))

store = Store(str(DATA_DIR / "agent.db"))
runner = Runner(store, DATA_DIR, max_parallel=int(os.environ.get("MAX_PARALLEL_JOBS", "2")))


def app_password():
    pw = os.environ.get("APP_PASSWORD") or store.get_setting("_generated_password")
    if not pw:
        pw = secrets.token_urlsafe(9)
        store.set_setting("_generated_password", pw)
    return pw


def make_token():
    return hmac.new(store.secret().encode(), app_password().encode(), hashlib.sha256).hexdigest()


def mask(key):
    return ("•••• " + key[-4:]) if key else ""


class Handler(BaseHTTPRequestHandler):
    server_version = "CodingAgent/1.0"

    # --- pomocnicze ---
    def log_message(self, fmt, *args):
        if "/api/health" not in (self.path or ""):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def err(self, msg, code=400):
        self.send_json({"error": msg}, code)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            raise ValueError("Za duże żądanie")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode() or "{}")

    def authed(self, query):
        token = ""
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        token = token or (query.get("t") or [""])[0]
        return bool(token) and hmac.compare_digest(token, make_token())

    def job_or_404(self, job_id):
        job = store.get_job(job_id)
        if not job:
            self.err("Nie znaleziono zadania", 404)
        return job

    # --- routing ---
    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_DELETE(self):
        self.route("DELETE")

    def route(self, method):
        try:
            u = urllib.parse.urlparse(self.path)
            path, query = u.path, urllib.parse.parse_qs(u.query)
            if method == "GET" and path in ("/", "/index.html"):
                return self.static("index.html")
            if method == "GET" and path in ("/manifest.json", "/icon.svg"):
                return self.static(path.lstrip("/"))
            if path == "/api/health":
                return self.send_json({"ok": True, "time": time.time()})
            if path == "/api/login" and method == "POST":
                data = self.body_json()
                if hmac.compare_digest(str(data.get("password", "")), app_password()):
                    return self.send_json({"token": make_token()})
                time.sleep(1.5)
                return self.err("Złe hasło", 401)
            if not path.startswith("/api/"):
                return self.err("Nie znaleziono", 404)
            if not self.authed(query):
                return self.err("Wymagane logowanie", 401)
            return self.api(method, path, query)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self.err(f"Błąd serwera: {e}", 500)
            except Exception:
                pass

    def static(self, name):
        p = BASE_DIR / "static" / name
        if not p.is_file():
            return self.err("Nie znaleziono", 404)
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def api(self, method, path, query):
        # ustawienia
        if path == "/api/settings":
            if method == "GET":
                return self.send_json({
                    "providers": {k: {"name": v["name"], "default_model": v["default_model"], "suggested": v["suggested"],
                                      "key": mask(get_api_key(store, k)),
                                      "key_from_env": bool(os.environ.get(v["key_env"])) and not store.get_setting(f"key_{k}")}
                                  for k, v in PROVIDERS.items()},
                    "default_provider": store.get_setting("default_provider", "gemini"),
                    "default_model_gemini": store.get_setting("default_model_gemini", ""),
                    "default_model_nvidia": store.get_setting("default_model_nvidia", ""),
                    "ntfy_topic": store.get_setting("ntfy_topic", ""),
                })
            data = self.body_json()
            for k in ("gemini", "nvidia"):
                if f"key_{k}" in data and data[f"key_{k}"] is not None:
                    store.set_setting(f"key_{k}", data[f"key_{k}"].strip())
            for k in ("default_provider", "default_model_gemini", "default_model_nvidia", "ntfy_topic"):
                if k in data:
                    store.set_setting(k, str(data[k]).strip())
            return self.send_json({"ok": True})

        if path == "/api/models" and method == "GET":
            prov = (query.get("provider") or ["gemini"])[0]
            if prov not in PROVIDERS:
                return self.err("Nieznany dostawca")
            try:
                return self.send_json({"models": list_models(store, prov)})
            except LLMError as e:
                return self.send_json({"models": [], "error": str(e)[:500]})

        # zadania
        if path == "/api/jobs":
            if method == "GET":
                jobs = store.list_jobs()
                for j in jobs:
                    j.pop("plan", None)
                    j["prompt"] = (j["prompt"] or "")[:200]
                return self.send_json({"jobs": jobs})
            if method == "POST":
                data = self.body_json()
                prompt = (data.get("prompt") or "").strip()
                provider = data.get("provider") or "gemini"
                if not prompt:
                    return self.err("Wpisz polecenie dla agenta")
                if provider not in PROVIDERS:
                    return self.err("Nieznany dostawca")
                if not get_api_key(store, provider):
                    return self.err(f"Brak klucza API {PROVIDERS[provider]['name']} — dodaj go w Ustawieniach")
                model = (data.get("model") or PROVIDERS[provider]["default_model"]).strip()
                title = (data.get("title") or "").strip() or prompt.splitlines()[0][:60]
                max_steps = max(5, min(int(data.get("max_steps") or 80), 1000))
                job_id = store.create_job(title, prompt, provider, model, max_steps)
                return self.send_json({"id": job_id})

        m = re.match(r"^/api/jobs/([\w\-]+)(/.*)?$", path)
        if m:
            job_id, sub = m.group(1), m.group(2) or ""
            job = self.job_or_404(job_id)
            if not job:
                return
            ws = runner.workspace(job_id)

            if sub == "" and method == "GET":
                job["plan"] = json.loads(job.get("plan") or "[]")
                job["active"] = runner.is_active(job_id)
                return self.send_json(job)
            if sub == "" and method == "DELETE":
                runner.stop(job_id)
                for _ in range(20):
                    if not runner.is_active(job_id):
                        break
                    time.sleep(0.5)
                store.delete_job(job_id)
                shutil.rmtree(ws.root, ignore_errors=True)
                return self.send_json({"ok": True})
            if sub == "/events" and method == "GET":
                after = int((query.get("after") or ["0"])[0])
                return self.send_json({"events": store.events(job_id, after)})
            if sub == "/stop" and method == "POST":
                runner.stop(job_id)
                return self.send_json({"ok": True})
            if sub == "/message" and method == "POST":
                if job["status"] in ("running", "queued") or runner.is_active(job_id):
                    return self.err("Agent jeszcze pracuje — poczekaj albo go zatrzymaj")
                data = self.body_json()
                text = (data.get("text") or "").strip()
                if not text:
                    return self.err("Pusta wiadomość")
                full = store.get_job(job_id, with_messages=True)
                msgs = json.loads(full["messages"] or "[]")
                upd = {}
                if data.get("provider") in PROVIDERS:
                    upd["provider"] = data["provider"]
                if data.get("model"):
                    upd["model"] = data["model"].strip()
                if not msgs:  # zadanie nigdy nie wystartowało — traktuj jako nowe polecenie
                    upd["prompt"] = (job["prompt"] + "\n\n" + text)
                else:
                    msgs.append({"role": "user", "content": text})
                    store.add_event(job_id, "user", text)
                store.update_job(job_id, messages=msgs, status="queued", run_steps=0, error=None, resumes=0, **upd)
                return self.send_json({"ok": True})
            if sub == "/files" and method == "GET":
                return self.send_json({"files": ws.all_files()})
            if sub == "/file" and method == "GET":
                rel = (query.get("path") or [""])[0]
                p = ws.path(rel)
                if not p.is_file():
                    return self.err("Brak pliku", 404)
                if (query.get("raw") or ["0"])[0] == "1":
                    body = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                    self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{urllib.parse.quote(p.name)}")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                size = p.stat().st_size
                raw = p.read_bytes()[:400_000]
                binary = b"\x00" in raw[:8000]
                return self.send_json({"path": rel, "size": size, "binary": binary,
                                       "content": "" if binary else raw.decode("utf-8", "replace"),
                                       "truncated": size > 400_000})
            if sub == "/download" and method == "GET":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for dirpath, dirnames, filenames in os.walk(ws.root):
                        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS - {".git"}]
                        for f in filenames:
                            fp = Path(dirpath) / f
                            z.write(fp, str(fp.relative_to(ws.root)))
                body = buf.getvalue()
                name = re.sub(r"[^\w\-]+", "_", job.get("title") or job_id)[:50] or job_id
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}.zip"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        return self.err("Nie znaleziono", 404)


def main():
    pw_source = "APP_PASSWORD" if os.environ.get("APP_PASSWORD") else "wygenerowane"
    print(f"▶ Agent programowania działa na porcie {PORT}  ·  dane: {DATA_DIR}", flush=True)
    print(f"▶ Hasło do panelu ({pw_source}): {app_password() if pw_source != 'APP_PASSWORD' else '(z APP_PASSWORD)'}", flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
