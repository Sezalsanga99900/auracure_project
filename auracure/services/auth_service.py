"""
services/auth_service.py
────────────────────────
Authentication and role-based access control for AuraEcho+.

Responsibility:
    Handle user registration, login, session management, and
    permission checking for the two clinical roles:
        • Doctor  — full access (diagnosis, AI, analytics, export)
        • Nurse   — restricted access (view records, vitals entry only)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Users stored in SQLite (auth.db) — separate from       │
    │  patient data for security isolation                    │
    │                                                         │
    │  Passwords hashed with bcrypt (never stored plain-text) │
    │                                                         │
    │  Sessions stored in Streamlit session_state + SQLite    │
    │  (survives page refresh, expires after SESSION_TTL)     │
    └─────────────────────────────────────────────────────────┘

Role permissions matrix:
    Permission              Doctor    Nurse
    ──────────────────────────────────────
    view_patient_data         ✅        ✅
    enter_vitals              ✅        ✅
    run_ai_diagnosis          ✅        ❌
    view_ai_results           ✅        ❌
    view_risk_scores          ✅        ✅
    export_data               ✅        ❌
    view_analytics            ✅        ✅
    manage_users              ✅        ❌
    delete_patient            ✅        ❌
    view_similar_cases        ✅        ✅

Public API:
    register_user(username, password, role, email) → bool
    login(username, password)                      → SessionToken | None
    logout(token)                                  → None
    get_current_user(token)                        → UserRecord | None
    has_permission(token, permission)              → bool
    require_permission(token, permission)          → None (raises if denied)
    get_all_users()                                → List[UserRecord]
    change_password(token, old_pw, new_pw)         → bool
    delete_user(admin_token, username)             → bool
    is_authenticated(token)                        → bool
"""

import hashlib
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from utils.constants import (
    AUTH_DB_PATH,
    ROLE_DOCTOR,
    ROLE_NURSE,
    SESSION_TTL_HOURS,
    ROLES,
    ROLE_PERMISSIONS,
)
from utils.helpers import get_logger
from utils.validators import validate_username, validate_password, validate_email

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Try bcrypt, fall back to sha256 if not installed
# ─────────────────────────────────────────────
try:
    import bcrypt
    _USE_BCRYPT = True
    logger.info("bcrypt available — using for password hashing")
except ImportError:
    _USE_BCRYPT = False
    logger.warning(
        "bcrypt not installed — using SHA-256 fallback. "
        "Install bcrypt for production: pip install bcrypt"
    )


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class UserRecord:
    """
    A single registered user in the system.

    Attributes
    ----------
    user_id    : str   UUID
    username   : str
    email      : str
    role       : str   "doctor" | "nurse"
    created_at : str   ISO-8601
    last_login : str   ISO-8601 or ""
    is_active  : bool
    """
    user_id:    str
    username:   str
    email:      str
    role:       str
    created_at: str
    last_login: str  = ""
    is_active:  bool = True

    @property
    def is_doctor(self) -> bool:
        return self.role == ROLE_DOCTOR

    @property
    def is_nurse(self) -> bool:
        return self.role == ROLE_NURSE

    @property
    def display_name(self) -> str:
        role_label = "Dr." if self.is_doctor else "Nurse"
        return f"{role_label} {self.username.title()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id":    self.user_id,
            "username":   self.username,
            "email":      self.email,
            "role":       self.role,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active":  self.is_active,
        }


@dataclass
class SessionToken:
    """
    An active user session.

    Attributes
    ----------
    token      : str   — cryptographically random hex string
    user_id    : str   — UUID of the authenticated user
    username   : str
    role       : str
    created_at : datetime
    expires_at : datetime
    """
    token:      str
    user_id:    str
    username:   str
    role:       str
    created_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.is_expired

    @property
    def ttl_minutes(self) -> int:
        """Minutes remaining before expiry."""
        remaining = self.expires_at - datetime.now(timezone.utc)
        return max(0, int(remaining.total_seconds() / 60))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token":      self.token,
            "user_id":    self.user_id,
            "username":   self.username,
            "role":       self.role,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ttl_minutes": self.ttl_minutes,
        }


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT    NOT NULL UNIQUE,
    username     TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT   NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'nurse',
    created_at   TEXT    NOT NULL,
    last_login   TEXT    NOT NULL DEFAULT '',
    is_active    INTEGER NOT NULL DEFAULT 1
);
"""

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token      TEXT    NOT NULL UNIQUE,
    user_id    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""

_CREATE_AUTH_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions(user_id);",
]


# ─────────────────────────────────────────────
# DB connection
# ─────────────────────────────────────────────

def _ensure_auth_dir() -> None:
    Path(AUTH_DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_auth_dir()
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_auth_db() -> None:
    """
    Create auth tables and seed a default admin doctor account if empty.
    Safe to call multiple times.
    """
    _ensure_auth_dir()
    with _get_conn() as conn:
        conn.execute(_CREATE_USERS_TABLE)
        conn.execute(_CREATE_SESSIONS_TABLE)
        for idx in _CREATE_AUTH_INDEXES:
            conn.execute(idx)
        conn.commit()

    # Seed default doctor account if no users exist
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    if count == 0:
        logger.info("No users found — seeding default admin doctor account")
        _seed_default_users()

    logger.info("Auth database initialised at %s", AUTH_DB_PATH)


def _seed_default_users() -> None:
    """
    Create default demo accounts for development/hackathon demo.

    Default credentials:
        Doctor: username=admin_doctor  password=Doctor@123
        Nurse:  username=nurse_demo    password=Nurse@123

    ⚠️  Change these immediately in production!
    """
    default_users = [
        {
            "username": "admin_doctor",
            "password": "Doctor@123",
            "role":     ROLE_DOCTOR,
            "email":    "doctor@auraecho.demo",
        },
        {
            "username": "nurse_demo",
            "password": "Nurse@123",
            "role":     ROLE_NURSE,
            "email":    "nurse@auraecho.demo",
        },
    ]
    for u in default_users:
        try:
            register_user(
                username=u["username"],
                password=u["password"],
                role=u["role"],
                email=u["email"],
                skip_validation=True,   # bypass strict validators for seed data
            )
            logger.info("Seeded user: %s (%s)", u["username"], u["role"])
        except Exception as exc:
            logger.warning("Could not seed user %s: %s", u["username"], exc)


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """
    Hash a plain-text password for storage.

    Uses bcrypt if available (preferred), otherwise SHA-256 + salt.
    """
    if _USE_BCRYPT:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    else:
        # SHA-256 + random salt (fallback)
        salt = secrets.token_hex(32)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"sha256:{salt}:{hashed}"


def _verify_password(plain: str, stored_hash: str) -> bool:
    """
    Verify a plain-text password against a stored hash.
    Handles both bcrypt and SHA-256 formats.
    """
    if stored_hash.startswith("sha256:"):
        _, salt, expected_hash = stored_hash.split(":", 2)
        actual = hashlib.sha256(f"{salt}{plain}".encode()).hexdigest()
        return secrets.compare_digest(actual, expected_hash)
    elif _USE_BCRYPT:
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False
    return False


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    d = dict(row)
    return UserRecord(
        user_id=d["user_id"],
        username=d["username"],
        email=d["email"],
        role=d["role"],
        created_at=d["created_at"],
        last_login=d.get("last_login", ""),
        is_active=bool(d.get("is_active", 1)),
    )


def _generate_token() -> str:
    """Generate a cryptographically secure 64-character session token."""
    return secrets.token_hex(32)   # 32 bytes = 64 hex chars


def _store_session(token: str, user_id: str) -> SessionToken:
    """Write a new session to the DB and return a SessionToken object."""
    now        = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_TTL_HOURS)

    # Load user details for the SessionToken object
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires_at.isoformat()),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()

    user = _row_to_user(row)
    return SessionToken(
        token=token,
        user_id=user_id,
        username=user.username,
        role=user.role,
        created_at=now,
        expires_at=expires_at,
    )


def _load_session(token: str) -> Optional[SessionToken]:
    """Load and return a SessionToken from the DB, or None if not found/expired."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT s.*, u.username, u.role FROM sessions s "
            "JOIN users u ON s.user_id = u.user_id "
            "WHERE s.token = ?",
            (token,),
        ).fetchone()

    if row is None:
        return None

    d          = dict(row)
    expires_at = datetime.fromisoformat(d["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    session = SessionToken(
        token=d["token"],
        user_id=d["user_id"],
        username=d["username"],
        role=d["role"],
        created_at=datetime.fromisoformat(d["created_at"]).replace(tzinfo=timezone.utc),
        expires_at=expires_at,
    )

    if session.is_expired:
        logger.debug("Session %s is expired — removing", token[:12])
        _delete_session(token)
        return None

    return session


def _delete_session(token: str) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def _purge_expired_sessions() -> int:
    """Delete all expired sessions. Returns count deleted."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        conn.commit()
        return cursor.rowcount


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def register_user(
    username:        str,
    password:        str,
    role:            str,
    email:           str,
    skip_validation: bool = False,
) -> bool:
    """
    Register a new user account.

    Parameters
    ----------
    username        : str — unique login name
    password        : str — plain-text (hashed immediately, never stored)
    role            : str — "doctor" | "nurse"
    email           : str — unique email address
    skip_validation : bool — bypass validators (only for seed data)

    Returns
    -------
    True on success.

    Raises
    ------
    ValueError  — if validation fails or username/email already exists
    """
    init_auth_db()

    # Validate inputs
    if not skip_validation:
        ok, err = validate_username(username)
        if not ok:
            raise ValueError(f"Invalid username: {err}")

        ok, err = validate_password(password)
        if not ok:
            raise ValueError(f"Invalid password: {err}")

        ok, err = validate_email(email)
        if not ok:
            raise ValueError(f"Invalid email: {err}")

    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}, got '{role}'")

    # Check uniqueness
    with _get_conn() as conn:
        existing_user  = conn.execute(
            "SELECT user_id FROM users WHERE username = ?", (username,)
        ).fetchone()
        existing_email = conn.execute(
            "SELECT user_id FROM users WHERE email = ?", (email,)
        ).fetchone()

    if existing_user:
        raise ValueError(f"Username '{username}' is already taken")
    if existing_email:
        raise ValueError(f"Email '{email}' is already registered")

    # Hash and store
    user_id       = str(uuid.uuid4())
    password_hash = _hash_password(password)
    now           = _now_iso()

    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO users
               (user_id, username, email, password_hash, role, created_at, last_login, is_active)
               VALUES (?, ?, ?, ?, ?, ?, '', 1)""",
            (user_id, username.lower(), email.lower(), password_hash, role, now),
        )
        conn.commit()

    logger.info("User registered: %s (%s)", username, role)
    return True


def login(username: str, password: str) -> Optional[SessionToken]:
    """
    Authenticate a user and create a session.

    Parameters
    ----------
    username : str
    password : str — plain-text password

    Returns
    -------
    SessionToken if credentials are valid and account is active.
    None if authentication fails.
    """
    init_auth_db()

    # Constant-time lookup (prevent username enumeration via timing)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()

    if row is None:
        # Still hash something to prevent timing attacks
        _verify_password(password, _hash_password("dummy_constant_time"))
        logger.warning("Login failed: unknown username '%s'", username)
        return None

    user = _row_to_user(row)
    stored_hash = dict(row)["password_hash"]

    if not _verify_password(password, stored_hash):
        logger.warning("Login failed: wrong password for '%s'", username)
        return None

    if not user.is_active:
        logger.warning("Login denied: account '%s' is inactive", username)
        return None

    # Create session
    token   = _generate_token()
    session = _store_session(token, user.user_id)

    # Update last_login timestamp
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE user_id = ?",
            (_now_iso(), user.user_id),
        )
        conn.commit()

    # Clean up old expired sessions periodically
    _purge_expired_sessions()

    logger.info("Login successful: %s (%s)", username, user.role)
    return session


def logout(token: str) -> None:
    """
    Invalidate a session token.

    Parameters
    ----------
    token : str — the session token to revoke
    """
    _delete_session(token)
    logger.info("Session revoked: %s...", token[:12])


def get_current_user(token: str) -> Optional[UserRecord]:
    """
    Return the UserRecord for a valid session token.

    Parameters
    ----------
    token : str

    Returns
    -------
    UserRecord if token is valid and not expired, None otherwise.
    """
    session = _load_session(token)
    if session is None:
        return None

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
            (session.user_id,),
        ).fetchone()

    if row is None:
        return None
    return _row_to_user(row)


def is_authenticated(token: str) -> bool:
    """Return True if the token is valid and not expired."""
    return _load_session(token) is not None


def get_session(token: str) -> Optional[SessionToken]:
    """Return the full SessionToken object for a token string."""
    return _load_session(token)


def has_permission(token: str, permission: str) -> bool:
    """
    Check whether the current user has a specific permission.

    Parameters
    ----------
    token      : str — active session token
    permission : str — permission name (from ROLE_PERMISSIONS in constants)

    Returns
    -------
    True if the user's role grants this permission.

    Example
    -------
    if has_permission(token, "run_ai_diagnosis"):
        run_diagnosis()
    """
    session = _load_session(token)
    if session is None:
        return False

    role_perms = ROLE_PERMISSIONS.get(session.role, {})
    return bool(role_perms.get(permission, False))


def require_permission(token: str, permission: str) -> None:
    """
    Raise PermissionError if the user does NOT have the given permission.

    Use as a guard at the top of restricted functions.

    Parameters
    ----------
    token      : str
    permission : str

    Raises
    ------
    PermissionError  — with a descriptive message
    ValueError       — if token is invalid/expired
    """
    session = _load_session(token)

    if session is None:
        raise ValueError("Session expired or invalid. Please log in again.")

    if not has_permission(token, permission):
        logger.warning(
            "Permission denied: user=%s role=%s permission=%s",
            session.username, session.role, permission,
        )
        raise PermissionError(
            f"Your role '{session.role}' does not have '{permission}' access. "
            "Please contact your administrator."
        )


def get_all_users() -> List[UserRecord]:
    """
    Return all registered users (for admin panel).

    Returns
    -------
    List[UserRecord] sorted by created_at ascending.
    """
    init_auth_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at ASC"
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def change_password(
    token:    str,
    old_pw:   str,
    new_pw:   str,
) -> bool:
    """
    Change the password for the currently logged-in user.

    Parameters
    ----------
    token  : str — active session token
    old_pw : str — current password (for verification)
    new_pw : str — new password

    Returns
    -------
    True on success, False if old_pw is wrong.

    Raises
    ------
    ValueError   — if new password fails validation
    ValueError   — if token is invalid
    """
    session = _load_session(token)
    if session is None:
        raise ValueError("Session invalid or expired")

    # Verify old password
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE user_id = ?",
            (session.user_id,),
        ).fetchone()

    if row is None or not _verify_password(old_pw, dict(row)["password_hash"]):
        return False

    # Validate new password
    ok, err = validate_password(new_pw)
    if not ok:
        raise ValueError(f"New password invalid: {err}")

    new_hash = _hash_password(new_pw)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (new_hash, session.user_id),
        )
        conn.commit()

    logger.info("Password changed for user: %s", session.username)
    return True


def deactivate_user(admin_token: str, username: str) -> bool:
    """
    Deactivate (soft-delete) a user account.
    Requires the caller to have 'manage_users' permission.

    Parameters
    ----------
    admin_token : str — must belong to a doctor with manage_users permission
    username    : str — account to deactivate

    Returns
    -------
    True on success.
    """
    require_permission(admin_token, "manage_users")

    with _get_conn() as conn:
        cursor = conn.execute(
            "UPDATE users SET is_active = 0 WHERE username = ? COLLATE NOCASE",
            (username,),
        )
        conn.commit()
        changed = cursor.rowcount > 0

    if changed:
        logger.info("User deactivated: %s", username)
    return changed


def get_user_permissions(token: str) -> Dict[str, bool]:
    """
    Return the full permissions dict for the current user's role.

    Returns
    -------
    dict mapping permission_name → True/False

    Example return:
    {
        "view_patient_data": True,
        "run_ai_diagnosis":  False,   # nurse
        "export_data":       False,   # nurse
        ...
    }
    """
    session = _load_session(token)
    if session is None:
        return {}
    return dict(ROLE_PERMISSIONS.get(session.role, {}))


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────
try:
    init_auth_db()
except Exception as _exc:
    logger.warning("Auth DB could not be initialised on import: %s", _exc)