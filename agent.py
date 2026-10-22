"""Silnik agenta: modele (Gemini / NVIDIA NIM przez API zgodne z OpenAI),
narzędzia do plików i terminala oraz pętla pracy działająca w tle."""
import fnmatch
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Dostawcy modeli
# ---------------------------------------------------------------------------
PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "base_url": os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-flash-latest",
        "suggested": ["gemini-flash-latest", "gemini-pro-latest", "gemini-flash-lite-latest"],
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "key_env": "NVIDIA_API_KEY",
        "default_model": "nvidia/nemotron-3-super-120b-a12b",
        "suggested": [
            "nvidia/nemotron-3-super-120b-a12b",
            "qwen/qwen3-coder-480b-a35b-instruct",
            "moonshotai/kimi-k2-instruct",
            "openai/gpt-oss-120b",
        ],
    },
}

MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "16384"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "300000"))
CMD_TIMEOUT_MAX = int(os.environ.get("CMD_TIMEOUT_MAX", "600"))
IGNORED_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".next", ".cache", "dist", "build", ".gradle"}
SECRET_ENV = {"GEMINI_API_KEY", "NVIDIA_API_KEY", "APP_PASSWORD", "NTFY_TOPIC"}


class LLMError(Exception):
    def __init__(self, msg, status=None, retryable=False, retry_after=None):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def get_api_key(store, provider):
    p = PROVIDERS[provider]
    return (store.get_setting(f"key_{provider}") or os.environ.get(p["key_env"]) or "").strip()


def _http_json(url, key, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:2000]
        ra = e.headers.get("Retry-After") if e.headers else None
        try:
            ra = float(ra) if ra else None
        except ValueError:
            ra = None
        raise LLMError(f"HTTP {e.code}: {text}", status=e.code,
                       retryable=e.code in (408, 409, 425, 429, 500, 502, 503, 504), retry_after=ra)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise LLMError(f"Błąd sieci: {e}", retryable=True)
    except json.JSONDecodeError as e:
        raise LLMError(f"Niepoprawna odpowiedź JSON: {e}", retryable=True)


def list_models(store, provider):
    key = get_api_key(store, provider)
    if not key:
        raise LLMError("Brak klucza API dla " + PROVIDERS[provider]["name"])
    res = _http_json(PROVIDERS[provider]["base_url"].rstrip("/") + "/models", key, timeout=30)
    ids = []
    for m in res.get("data", []):
        mid = m.get("id", "")
        if mid.startswith("models/"):
            mid = mid[7:]
        if provider == "gemini" and "gemini" not in mid:
            continue
        if any(bad in mid for bad in ("embed", "rerank", "guard", "parse", "tts", "image", "vision-only", "clip")):
            continue
        ids.append(mid)
    return sorted(set(ids))


def chat(store, provider, model, messages, tools):
    key = get_api_key(store, provider)
    if not key:
        raise LLMError("Brak klucza API dla " + PROVIDERS[provider]["name"] + " — dodaj go w Ustawieniach.")
    url = PROVIDERS[provider]["base_url"].rstrip("/") + "/chat/completions"
    max_tokens = MAX_OUTPUT_TOKENS
    while True:
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens}
        try:
            return _http_json(url, key, body)
        except LLMError as e:
            # część modeli NIM ma niższy limit tokenów wyjścia — zmniejszamy i próbujemy ponownie
            if e.status == 400 and "token" in str(e).lower() and max_tokens > 2048:
                max_tokens //= 2
                continue
            raise


# ---------------------------------------------------------------------------
# Narzędzia
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "update_plan",
        "description": "Ustaw/aktualizuj listę kroków planu (widoczną dla użytkownika). Wywołaj na początku i odhaczaj kroki w trakcie.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "done": {"type": "boolean"}}, "required": ["text", "done"]}}},
            "required": ["items"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Pokaż drzewo plików i folderów projektu (ścieżki względne). Pomija node_modules, .git itp.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder, domyślnie '.'"},
            "max_depth": {"type": "integer", "description": "Głębokość, domyślnie 4"}}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Odczytaj plik tekstowy. Dla dużych plików podaj zakres linii.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Zapisz plik (tworzy brakujące foldery, nadpisuje). Zapisuj PEŁNĄ treść. Bardzo duże pliki zapisuj w częściach z append=true.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "append": {"type": "boolean", "description": "Dopisz na końcu zamiast nadpisywać"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Zamień dokładny fragment tekstu w pliku na nowy. old_text musi wystąpić dokładnie raz (chyba że replace_all=true).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"}},
            "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "make_dir",
        "description": "Utwórz folder (z rodzicami).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "move_path",
        "description": "Przenieś lub zmień nazwę pliku/folderu.",
        "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                       "required": ["src", "dst"]}}},
    {"type": "function", "function": {
        "name": "delete_path",
        "description": "Usuń plik lub folder (rekurencyjnie).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Szukaj wyrażenia regularnego w plikach projektu. Zwraca plik:linia: treść.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "glob": {"type": "string", "description": "np. *.py"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Uruchom polecenie powłoki (bash) w folderze projektu, np. instalacja zależności, testy, kompilacja, git. "
                       "NIE uruchamiaj serwerów działających bez końca — do sprawdzenia użyj np. `timeout 10 ...`.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Sekundy, domyślnie 120"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Zakończ zadanie, gdy WSZYSTKO jest gotowe i sprawdzone. Podaj podsumowanie dla użytkownika (co powstało, jak uruchomić).",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel):
        rel = (rel or ".").strip()
        if rel.startswith("/workspace"):
            rel = rel[len("/workspace"):] or "."
        p = (self.root / rel.lstrip("/")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"Ścieżka poza projektem jest zabroniona: {rel}")
        return p

    def rel(self, p):
        return str(Path(p).relative_to(self.root)) or "."

    def tree(self, rel=".", max_depth=4, limit=600):
        base = self.path(rel)
        if not base.exists():
            raise FileNotFoundError(f"Nie istnieje: {rel}")
        lines, count = [], 0

        def walk(d, depth, prefix):
            nonlocal count
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return
            for e in entries:
                if count >= limit:
                    return
                count += 1
                if e.is_dir():
                    if e.name in IGNORED_DIRS:
                        lines.append(f"{prefix}{e.name}/  (pominięto)")
                        continue
                    lines.append(f"{prefix}{e.name}/")
                    if depth < max_depth:
                        walk(e, depth + 1, prefix + "  ")
                else:
                    lines.append(f"{prefix}{e.name}  ({e.stat().st_size} B)")

        walk(base, 1, "")
        if count >= limit:
            lines.append(f"... (ucięto po {limit} wpisach)")
        return "\n".join(lines) or "(pusty folder)"

    def all_files(self, limit=3000):
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for f in sorted(filenames):
                fp = Path(dirpath) / f
                try:
                    out.append({"path": self.rel(fp), "size": fp.stat().st_size})
                except OSError:
                    pass
                if len(out) >= limit:
                    return out
        return out


def _clip(text, limit=15000):
    if len(text) <= limit:
        return text
    head = limit // 5
    return text[:head] + f"\n\n... [pominięto {len(text) - limit} znaków] ...\n\n" + text[-(limit - head):]


def _safe_env(ws_root):
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env.update({"CI": "1", "npm_config_yes": "true", "PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "TERM": "dumb", "PROJECT_DIR": str(ws_root)})
    return env


def run_tool(ws, name, args, job_ctx):
    """Wykonuje narzędzie; zwraca tekst wyniku dla modelu."""
    if name == "update_plan":
        items = args.get("items") or []
        job_ctx["plan"] = [{"text": str(i.get("text", "")), "done": bool(i.get("done"))} for i in items if isinstance(i, dict)]
        return "Plan zaktualizowany."
    if name == "list_files":
        return ws.tree(args.get("path") or ".", int(args.get("max_depth") or 4))
    if name == "read_file":
        p = ws.path(args["path"])
        if not p.is_file():
            return f"BŁĄD: plik nie istnieje: {args['path']}"
        if p.stat().st_size > 5_000_000:
            return "BŁĄD: plik jest za duży (>5 MB)."
        text = p.read_text("utf-8", errors="replace")
        lines = text.splitlines()
        s, e = args.get("start_line"), args.get("end_line")
        if s or e:
            s = max(int(s or 1), 1)
            e = min(int(e or len(lines)), len(lines))
            chunk = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
            return _clip(f"[{args['path']} linie {s}-{e} z {len(lines)}]\n{chunk}", 40000)
        return _clip(f"[{args['path']} — {len(lines)} linii]\n{text}", 40000)
    if name == "write_file":
        p = ws.path(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        content = args.get("content", "")
        if args.get("append"):
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Dopisano {len(content)} znaków do {args['path']} (rozmiar: {p.stat().st_size} B)."
        p.write_text(content, encoding="utf-8")
        return f"Zapisano {args['path']} ({len(content)} znaków, {content.count(chr(10)) + 1} linii)."
    if name == "edit_file":
        p = ws.path(args["path"])
        if not p.is_file():
            return f"BŁĄD: plik nie istnieje: {args['path']}"
        text = p.read_text("utf-8", errors="replace")
        old, new = args.get("old_text", ""), args.get("new_text", "")
        n = text.count(old) if old else 0
        if n == 0:
            return "BŁĄD: nie znaleziono old_text w pliku. Odczytaj plik i spróbuj ponownie z dokładnym fragmentem."
        if n > 1 and not args.get("replace_all"):
            return f"BŁĄD: old_text występuje {n} razy. Podaj dłuższy, unikalny fragment albo ustaw replace_all=true."
        text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        return f"Zmieniono {args['path']} ({n if args.get('replace_all') else 1} zamian)."
    if name == "make_dir":
        ws.path(args["path"]).mkdir(parents=True, exist_ok=True)
        return f"Utworzono folder {args['path']}."
    if name == "move_path":
        src, dst = ws.path(args["src"]), ws.path(args["dst"])
        if not src.exists():
            return f"BŁĄD: nie istnieje: {args['src']}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Przeniesiono {args['src']} -> {args['dst']}."
    if name == "delete_path":
        p = ws.path(args["path"])
        if p == ws.root:
            return "BŁĄD: nie można usunąć całego projektu."
        if not p.exists():
            return f"Nie istnieje: {args['path']}"
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        return f"Usunięto {args['path']}."
    if name == "search_files":
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            return f"BŁĄD: niepoprawny regex: {e}"
        base, glob, hits = ws.path(args.get("path") or "."), args.get("glob"), []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for f in filenames:
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                fp = Path(dirpath) / f
                try:
                    if fp.stat().st_size > 1_000_000:
                        continue
                    for i, line in enumerate(fp.read_text("utf-8", errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{ws.rel(fp)}:{i}: {line.strip()[:200]}")
                            if len(hits) >= 200:
                                return "\n".join(hits) + "\n... (limit 200 wyników)"
                except OSError:
                    continue
        return "\n".join(hits) or "Brak wyników."
    if name == "run_command":
        cmd = args["command"]
        timeout = max(1, min(int(args.get("timeout") or 120), CMD_TIMEOUT_MAX))
        proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(ws.root), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=_safe_env(ws.root),
                                start_new_session=True)
        job_ctx["proc"] = proc
        try:
            out, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            out, _ = proc.communicate()
            code, timed_out = -9, True
        finally:
            job_ctx["proc"] = None
        out = (out or b"").decode("utf-8", "replace")
        status = f"PRZEKROCZONO CZAS ({timeout}s), proces zabity" if timed_out else f"kod wyjścia {code}"
        return f"[{status}]\n{_clip(out, 12000) or '(brak wyjścia)'}"
    return f"BŁĄD: nieznane narzędzie {name}"


# ---------------------------------------------------------------------------
# Pętla agenta
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous senior software engineer working ALONE on a server, with no human watching.
The user gave you a task from their phone and closed the browser; you must complete it end-to-end without asking questions.

Environment:
- Your project folder is the workspace root. All tool paths are RELATIVE to it (e.g. "src/app.js").
- Linux shell via run_command (bash). Available tools: {tooling}.
- Commands time out (default 120 s, max {cmd_max} s). Never start servers/watchers that run forever; to smoke-test use e.g. `timeout 10 node server.js`.

How to work:
1. Start with update_plan (5-12 concrete steps). Keep it updated (mark done) as you progress.
2. Explore with list_files/read_file when the workspace already has files.
3. Write COMPLETE, working files — no placeholders, no "TODO: implement", no truncated code.
   For very large files write them in parts with write_file(append=true).
4. Verify your work: install dependencies, run tests/linters/build, run the program briefly. Fix every error you see.
5. Always add a README.md with what it is and exact steps to run it.
6. If something is ambiguous, choose sensible defaults and mention them in the summary.
7. When everything works, call finish(summary). Write the summary in the same language as the user's request
   (Polish if the user writes Polish): what was built, file structure overview, how to run it, known limitations.
Never stop with plain text before calling finish — keep using tools until the task is done."""


def _tooling_info():
    found = [t for t in ("python3", "pip", "node", "npm", "git", "gcc", "java", "go", "cargo") if shutil.which(t)]
    return ", ".join(found) or "basic shell only"


def repair_history(messages):
    """Każde wywołanie narzędzia musi mieć odpowiedź (ważne po restarcie serwera w trakcie kroku)."""
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            answered = set()
            while i < len(messages) and messages[i].get("role") == "tool":
                answered.add(messages[i].get("tool_call_id"))
                out.append(messages[i])
                i += 1
            for tc in m["tool_calls"]:
                if tc.get("id") not in answered:
                    out.append({"role": "tool", "tool_call_id": tc.get("id"),
                                "name": tc.get("function", {}).get("name", ""),
                                "content": "PRZERWANO (restart serwera) — sprawdź stan plików i kontynuuj."})
            _ = ids
    return out


def compact(messages):
    """Ogranicza rozmiar historii: skraca stare wyniki narzędzi i treść zapisanych plików."""
    def size():
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    if size() <= MAX_CONTEXT_CHARS:
        return
    keep_tail = 14
    end = max(2, len(messages) - keep_tail)
    for m in messages[2:end]:  # najpierw duże argumenty write_file/edit_file
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") in ("write_file", "edit_file") and len(fn.get("arguments", "")) > 600:
                try:
                    a = json.loads(fn["arguments"])
                    for k in ("content", "old_text", "new_text"):
                        if k in a and len(a[k]) > 200:
                            a[k] = a[k][:120] + " …[pominięto — treść jest już w pliku]"
                    fn["arguments"] = json.dumps(a, ensure_ascii=False)
                except Exception:
                    pass
    if size() <= MAX_CONTEXT_CHARS:
        return
    for m in messages[2:end]:
        if m.get("role") == "tool" and len(m.get("content") or "") > 400:
            m["content"] = m["content"][:300] + "\n…[skrócono stary wynik — uruchom/odczytaj ponownie, jeśli potrzebny]"


def _short(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def _describe_call(name, a):
    if name in ("write_file",):
        return f"{'dopisuje' if a.get('append') else 'zapisuje'} {a.get('path')} ({len(a.get('content', ''))} zn.)"
    if name in ("read_file", "edit_file", "make_dir", "delete_path"):
        return f"{name} {a.get('path')}"
    if name == "move_path":
        return f"move {a.get('src')} → {a.get('dst')}"
    if name == "run_command":
        return f"$ {_short(a.get('command', ''), 300)}"
    if name == "search_files":
        return f"szuka /{a.get('pattern')}/"
    if name == "list_files":
        return f"lista plików {a.get('path') or '.'}"
    if name == "update_plan":
        return "aktualizuje plan"
    return f"{name} {_short(json.dumps(a, ensure_ascii=False))}"


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)  # czasem modele otaczają JSON tekstem
        if m:
            return json.loads(m.group(0))
        raise


class Runner:
    def __init__(self, store, data_dir, max_parallel=2):
        self.store = store
        self.ws_dir = Path(data_dir) / "workspaces"
        self.ws_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.active = {}  # job_id -> ctx
        self.lock = threading.Lock()
        self._last_ping = 0
        self._recover()
        threading.Thread(target=self._dispatch, daemon=True, name="dispatcher").start()

    def workspace(self, job_id):
        return Workspace(self.ws_dir / job_id)

    def _recover(self):
        # zadania przerwane restartem serwera wznawiamy automatycznie (max 3 razy)
        for j in self.store.q("SELECT id, resumes FROM jobs WHERE status='running'"):
            if (j["resumes"] or 0) < 3:
                self.store.update_job(j["id"], status="queued", resumes=(j["resumes"] or 0) + 1)
                self.store.add_event(j["id"], "info", "Serwer został zrestartowany — wznawiam pracę.")
            else:
                self.store.update_job(j["id"], status="interrupted", error="Zbyt wiele restartów serwera.")
                self.store.add_event(j["id"], "error", "Przerwano: zbyt wiele restartów serwera.")

    def stop(self, job_id):
        with self.lock:
            ctx = self.active.get(job_id)
        if ctx:
            ctx["stop"] = True
            proc = ctx.get("proc")
            if proc:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        else:
            job = self.store.get_job(job_id)
            if job and job["status"] == "queued":
                self.store.update_job(job_id, status="stopped")
                self.store.add_event(job_id, "info", "Zatrzymano przed startem.")

    def is_active(self, job_id):
        with self.lock:
            return job_id in self.active

    def _dispatch(self):
        while True:
            try:
                with self.lock:
                    free = self.max_parallel - len(self.active)
                if free > 0:
                    for row in self.store.q("SELECT id FROM jobs WHERE status='queued' ORDER BY updated LIMIT ?", (free,)):
                        jid = row["id"]
                        with self.lock:
                            if jid in self.active:
                                continue
                            ctx = {"stop": False, "proc": None, "plan": None}
                            self.active[jid] = ctx
                        self.store.update_job(jid, status="running", error=None)
                        threading.Thread(target=self._run_safe, args=(jid, ctx), daemon=True, name=f"job-{jid}").start()
                self._keepalive()
            except Exception:
                traceback.print_exc()
            time.sleep(1)

    def _keepalive(self):
        """Na darmowych hostingach (np. Render free) serwer usypia bez ruchu — pingujemy sami siebie w trakcie pracy."""
        url = os.environ.get("KEEPALIVE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
        with self.lock:
            busy = bool(self.active)
        if not url or not busy or time.time() - self._last_ping < 240:
            return
        self._last_ping = time.time()

        def ping():
            try:
                urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=20).read()
            except Exception:
                pass
        threading.Thread(target=ping, daemon=True).start()

    def _notify(self, job, text):
        topic = self.store.get_setting("ntfy_topic") or os.environ.get("NTFY_TOPIC")
        if not topic:
            return
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{topic.strip()}", data=text.encode("utf-8"), method="POST")
            req.add_header("Title", f"Agent: {job.get('title') or job['id']}"[:200].encode("utf-8").decode("latin-1", "ignore"))
            urllib.request.urlopen(req, timeout=15).read()
        except Exception:
            pass

    def _run_safe(self, job_id, ctx):
        try:
            self._run(job_id, ctx)
        except Exception as e:
            traceback.print_exc()
            self.store.update_job(job_id, status="error", error=str(e)[:2000])
            self.store.add_event(job_id, "error", f"Nieoczekiwany błąd: {e}")
            job = self.store.get_job(job_id) or {"id": job_id}
            self._notify(job, f"❌ Błąd: {str(e)[:300]}")
        finally:
            with self.lock:
                self.active.pop(job_id, None)

    def _call_llm(self, job_id, ctx, provider, model, messages):
        delays = [3, 6, 12, 25, 45, 60, 90, 120]
        attempt = 0
        while True:
            if ctx["stop"]:
                return None
            try:
                return chat(self.store, provider, model, messages, TOOLS)
            except LLMError as e:
                if not e.retryable or attempt >= len(delays):
                    raise
                wait = min(e.retry_after or delays[attempt], 180)
                attempt += 1
                self.store.add_event(job_id, "info", f"Model chwilowo niedostępny ({_short(str(e), 120)}). Ponowienie za {int(wait)} s…")
                for _ in range(int(wait)):
                    if ctx["stop"]:
                        return None
                    time.sleep(1)

    def _run(self, job_id, ctx):
        store = self.store
        job = store.get_job(job_id, with_messages=True)
        ws = self.workspace(job_id)
        provider, model = job["provider"], job["model"]
        if provider not in PROVIDERS:
            raise ValueError(f"Nieznany dostawca: {provider}")
        messages = json.loads(job["messages"] or "[]")
        ctx["plan"] = json.loads(job["plan"] or "[]")
        if not messages:
            system = SYSTEM_PROMPT.format(tooling=_tooling_info(), cmd_max=CMD_TIMEOUT_MAX)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": job["prompt"]}]
            store.add_event(job_id, "user", job["prompt"])
            store.update_job(job_id, messages=messages)
        messages = repair_history(messages)
        store.add_event(job_id, "info", f"Start pracy · {PROVIDERS[provider]['name']} · {model}")

        steps, run_steps = job["steps"] or 0, job["run_steps"] or 0
        max_steps = job["max_steps"] or 80
        nudges = 0

        def save(**extra):
            store.update_job(job_id, messages=messages, plan=ctx["plan"] or [], steps=steps, run_steps=run_steps, **extra)

        while True:
            if ctx["stop"]:
                save(status="stopped")
                store.add_event(job_id, "info", "Zatrzymano na prośbę użytkownika.")
                return
            if run_steps >= max_steps:
                save(status="paused")
                store.add_event(job_id, "info", f"Osiągnięto limit {max_steps} kroków. Napisz „kontynuuj”, aby agent pracował dalej.")
                self._notify(job, f"⏸ Limit kroków ({max_steps}). Wyślij 'kontynuuj', by dokończyć.")
                return

            compact(messages)
            resp = self._call_llm(job_id, ctx, provider, model, messages)
            if resp is None:
                continue  # stop
            usage = resp.get("usage") or {}
            store.add_tokens(job_id, usage.get("prompt_tokens"), usage.get("completion_tokens"))
            choices = resp.get("choices") or []
            if not choices:
                raise LLMError(f"Pusta odpowiedź modelu: {_short(json.dumps(resp), 300)}")
            choice = choices[0]
            msg = choice.get("message") or {}
            steps += 1
            run_steps += 1

            content = msg.get("content")
            if isinstance(content, list):  # niektóre API zwracają listę części
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                tc.setdefault("type", "function")
                if not tc.get("id"):
                    tc["id"] = "call_" + secrets.token_hex(6)

            assistant = {"role": "assistant", "content": content if content else (None if tool_calls else "")}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            save()  # zapis przed wykonaniem narzędzi — po restarcie historia zostanie naprawiona i praca wznowiona
            if content and content.strip():
                store.add_event(job_id, "assistant", content.strip())

            if not tool_calls:
                if choice.get("finish_reason") == "length":
                    messages.append({"role": "user", "content": "Your reply was cut off (output limit). Continue, and write big files in smaller parts with write_file(append=true)."})
                    save()
                    continue
                nudges += 1
                if nudges >= 2:
                    summary = (content or "").strip() or "Agent zakończył pracę."
                    save(status="done", summary=summary)
                    store.add_event(job_id, "done", summary)
                    self._notify(job, "✅ Gotowe: " + _short(summary, 300))
                    return
                messages.append({"role": "user", "content": "Continue working autonomously using the tools. When everything is complete and verified, call the finish tool with a summary."})
                save()
                continue
            nudges = 0

            finished_summary = None
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = _parse_args(fn.get("arguments"))
                except Exception:
                    result = ("BŁĄD: argumenty narzędzia nie są poprawnym JSON-em (prawdopodobnie ucięte przez limit długości). "
                              "Zapisuj duże pliki w mniejszych częściach (write_file z append=true).")
                    store.add_event(job_id, "tool_error", f"{name}: niepoprawne argumenty")
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": result})
                    continue
                if name == "finish":
                    finished_summary = str(args.get("summary") or "Gotowe.")
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": "OK"})
                    continue
                store.add_event(job_id, "tool", _describe_call(name, args), {"name": name})
                t0 = time.time()
                try:
                    result = run_tool(ws, name, args, ctx)
                except Exception as e:
                    result = f"BŁĄD: {type(e).__name__}: {e}"
                is_err = result.startswith("BŁĄD") or (name == "run_command" and not result.startswith("[kod wyjścia 0]"))
                store.add_event(job_id, "tool_error" if is_err else "result", _clip(result, 4000),
                                {"name": name, "secs": round(time.time() - t0, 1)})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": result})
                if ctx["stop"]:
                    break
            messages[:] = repair_history(messages)

            if finished_summary is not None:
                save(status="done", summary=finished_summary)
                store.add_event(job_id, "done", finished_summary)
                self._notify(job, "✅ Gotowe: " + _short(finished_summary, 300))
                return
            save()
