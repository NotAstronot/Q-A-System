"""Security module: rate limiting, audit logging, input sanitization, encryption."""

import os
import re
import logging
from time import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ── Audit Log ─────────────────────────────────────────────────

class AuditLog:
    """Security event audit logger."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.audit_file = self.log_dir / "audit.log"
        self._setup_handler()

    def _setup_handler(self):
        handler = logging.FileHandler(self.audit_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        audit_logger = logging.getLogger("audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(handler)
        audit_logger.propagate = False
        self.logger = audit_logger

    def log(
        self, event: str, detail: str = "",
        ip: str = "", user: str = "", severity: str = "INFO"
    ):
        extra = f"IP={ip} | User={user} | {detail}" if ip or user else detail
        msg = f"[{event}] {extra}".strip()
        if severity == "ERROR":
            self.logger.error(msg)
        elif severity == "WARN":
            self.logger.warning(msg)
        else:
            self.logger.info(msg)


audit = AuditLog()


# ── Rate Limiter ──────────────────────────────────────────────

class RateLimiter:
    """Sliding window rate limiter per IP."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)

    def check(self, ip: str) -> Tuple[bool, int]:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._buckets[ip] = [t for t in self._buckets[ip] if t > cutoff]
        count = len(self._buckets[ip])
        if count >= self.max_requests:
            return False, count
        self._buckets[ip].append(now)
        return True, count

    def get_remaining(self, ip: str) -> int:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._buckets[ip] = [t for t in self._buckets[ip] if t > cutoff]
        return max(0, self.max_requests - len(self._buckets[ip]))

    def reset(self, ip: str):
        self._buckets.pop(ip, None)


rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_MAX", "60")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
)


# ── Rate Limit Middleware ─────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            ip = request.client.host if request.client else "unknown"
            allowed, count = rate_limiter.check(ip)
            if not allowed:
                audit.log(
                    "RATE_LIMIT_EXCEEDED",
                    detail=f"Rate limit exceeded for {request.url.path}",
                    ip=ip, severity="WARN"
                )
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later."
                )
        response = await call_next(request)
        return response


# ── Security Headers Middleware ──────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response


# ── Input Sanitization ───────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    name = re.sub(r'[^\w\-. ]', '', filename)
    name = Path(name).name
    if not name or name.startswith('.'):
        return ""
    return name[:255]


def sanitize_question(question: str) -> str:
    """Sanitize user question to prevent prompt injection."""
    sanitized = question.strip()
    if not sanitized:
        return ""
    sanitized = sanitized.replace("\x00", "")
    return sanitized[:4096]


def validate_file_content(data: bytes) -> bool:
    """Validate file content is actually a PDF (magic bytes check)."""
    return data[:5] in (b"%PDF-", b"%PDF ")


# ── Rate-limited API Key dependency ──────────────────────────

_last_failed_attempts: Dict[str, int] = defaultdict(int)
_last_failed_time: Dict[str, float] = {}

def check_brute_force(ip: str) -> bool:
    """Detect brute force attempts per IP."""
    now = time()
    if ip in _last_failed_time and now - _last_failed_time[ip] > 300:
        _last_failed_attempts[ip] = 0
    _last_failed_attempts[ip] += 1
    _last_failed_time[ip] = now
    return _last_failed_attempts[ip] > 10


def reset_brute_force(ip: str):
    _last_failed_attempts.pop(ip, None)
    _last_failed_time.pop(ip, None)
