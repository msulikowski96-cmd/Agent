"""Trwały magazyn: zadania, zdarzenia (log) i ustawienia w SQLite."""
import json
import os
import secrets
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  title TEXT,
  prompt TEXT,
  provider TEXT,
  model TEXT,
  status TEXT,
  created REAL,
  updated REAL,
  max_steps INTEGER,
  steps INTEGER DEFAULT 0,
  run_steps INTEGER DEFAULT 0,
  messages TEXT,
  plan TEXT,
  summary TEXT,
  error TEXT,
  resumes INTEGER DEFAULT 0,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT,
  ts REAL,
  kind TEXT,
  text TEXT,
  data TEXT
);
CREATE INDEX IF NOT EXISTS ev_job ON events(job_id, id);
CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
"""

JOB_FIELDS = (
    "id", "title", "prompt", "provider", "model", "status", "created", "updated",
    "max_steps", "steps", "run_steps", "plan", "summary", "error", "resumes",
    "tokens_in", "tokens_out",
)


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            self.db.commit()

    # --- niskopoziomowe ---
    def q(self, sql, args=(), one=False):
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
        if one:
            return dict(rows[0]) if rows else None
        return [dict(r) for r in rows]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # --- ustawienia ---
    def get_setting(self, key, default=None):
        row = self.q("SELECT v FROM settings WHERE k=?", (key,), one=True)
        return row["v"] if row and row["v"] is not None else default

    def set_setting(self, key, value):
        self.x("INSERT INTO settings(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))

    def secret(self):
        s = self.get_setting("_secret")
        if not s:
            s = secrets.token_hex(32)
            self.set_setting("_secret", s)
        return s

    # --- zadania ---
    def create_job(self, title, prompt, provider, model, max_steps):
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        now = time.time()
        self.x(
            "INSERT INTO jobs(id,title,prompt,provider,model,status,created,updated,max_steps,messages,plan)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, title, prompt, provider, model, "queued", now, now, max_steps, "[]", "[]"),
        )
        return job_id

    def get_job(self, job_id, with_messages=False):
        cols = ",".join(JOB_FIELDS) + (",messages" if with_messages else "")
        return self.q(f"SELECT {cols} FROM jobs WHERE id=?", (job_id,), one=True)

    def list_jobs(self, limit=100):
        cols = ",".join(JOB_FIELDS)
        return self.q(f"SELECT {cols} FROM jobs ORDER BY created DESC LIMIT ?", (limit,))

    def update_job(self, job_id, **fields):
        if not fields:
            return
        fields["updated"] = time.time()
        for k, v in list(fields.items()):
            if isinstance(v, (list, dict)):
                fields[k] = json.dumps(v, ensure_ascii=False)
        sets = ",".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def add_tokens(self, job_id, tin, tout):
        self.x("UPDATE jobs SET tokens_in=tokens_in+?, tokens_out=tokens_out+? WHERE id=?", (tin or 0, tout or 0, job_id))

    def delete_job(self, job_id):
        self.x("DELETE FROM events WHERE job_id=?", (job_id,))
        self.x("DELETE FROM jobs WHERE id=?", (job_id,))

    # --- zdarzenia (log widoczny w panelu) ---
    def add_event(self, job_id, kind, text, data=None):
        return self.x(
            "INSERT INTO events(job_id,ts,kind,text,data) VALUES(?,?,?,?,?)",
            (job_id, time.time(), kind, text, json.dumps(data, ensure_ascii=False) if data is not None else None),
        )

    def events(self, job_id, after=0, limit=500):
        return self.q(
            "SELECT id,ts,kind,text,data FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, after, limit),
        )
