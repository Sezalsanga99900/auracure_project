# =============================================================================
# services/auth_service.py
# AuraEcho+ — Authentication & Authorization Service
#
# Responsibility:
#     Manage user authentication, role-based access control, session management,
#     and account lockout policies. Uses a separate SQLite database for auth.
#
# Security Features:
#     • bcrypt password hashing with salt
#     • Account lockout after MAX_LOGIN_ATTEMPTS
#     • Session expiry after SESSION_TTL_HOURS
#     • Role-based permission checks
#     • Secure session tokens
#
# Public API:
#     init_auth_db()                    → None
#     create_user(username, password, role) → bool
#     authenticate(username, password)  → dict | None
#     check_permission(role, permission) → bool
#     get_user(username)                → dict | None
#     update_user_role(username, role)  → bool
#     delete_user(username)             → bool
#     list_users()                      → list[dict]
#     create_session(user)              → str (session_token)
#     validate_session(token)           → dict | None
#     clear_session(token)              → None
#     create_default_admin()            → bool
# =============================================================================

import os
import sqlite3
import threading
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from utils.constants import (
    AUTH_DB_PATH,
    ROLES,
    ROLE_PERMISSIONS,
    ROLE_DOCTOR,
    ROLE_NURSE,
    ROLE_ADMIN,
    SESSION_TTL_HOURS,
    PASSWORD_MIN_LENGTH,
    MAX_LOGIN_ATTEMPTS,
    LOCKOUT_DURATION_MIN,
    JWT_ALGORITHM,
)
from utils.helpers import get_logger, ensure_dir, now_str, mask_key
from utils.validators import (
    validate_username,
    validate_password,
    validate_role,
    validate_login_attempt,
)

logger = get_logger(__name__)

# Thread lock for database operations
_db_lock = threading.Lock()

# In-memory session store (for Streamlit compatibility)
# In production, consider Redis or database-backed sessions
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """
    Hash password using bcrypt (preferred) or PBKDF2 fallback.
    Returns hashed password string.
    """
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    else:
        # Fallback: PBKDF2 with SHA-256
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        )
        return f"pbkdf2${salt}${dk.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against stored hash.
    Supports both bcrypt and PBKDF2 formats.
    """
    try:
        if hashed.startswith("pbkdf2$"):
            # PBKDF2 format
            _, salt, stored_dk = hashed.split("$")
            dk = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                100000,
            )
            return secrets.compare_digest(dk.hex(), stored_dk)
        elif BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        else:
            logger.error("Cannot verify bcrypt hash — bcrypt not installed")
            return False
    except Exception as exc:
        logger.error("Password verification error: %s", exc)
        return False


# ─────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Get database connection with row factory."""
    ensure_dir(os.path.dirname(AUTH_DB_PATH))
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─────────────────────────────────────────────
# Schema initialization
# ─────────────────────────────────────────────

def init_auth_db() -> None:
    """
    Create auth database tables if they don't exist.
    Call this once at app startup.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                password_hash   TEXT NOT NULL,
                role            TEXT NOT NULL,
                full_name       TEXT,
                email           TEXT,
                failed_attempts INTEGER DEFAULT 0,
                locked_until    TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                last_login      TEXT
            )
        """)

        # Sessions table (persistent sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT PRIMARY KEY,
                username    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                is_active   INTEGER DEFAULT 1,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)")

        conn.commit()
        conn.close()
        logger.info("Auth database initialized at %s", AUTH_DB_PATH)


# ─────────────────────────────────────────────
# User operations
# ─────────────────────────────────────────────

def create_user(
    username:   str,
    password:   str,
    role:       str,
    full_name:  Optional[str] = None,
    email:      Optional[str] = None,
) -> bool:
    """
    Create a new user account.

    Returns True on success, False on validation failure or duplicate.
    """
    # Validate inputs
    ok, err = validate_username(username)
    if not ok:
        logger.warning("create_user: invalid username — %s", err)
        return False

    ok, err = validate_password(password)
    if not ok:
        logger.warning("create_user: invalid password — %s", err)
        return False

    ok, err = validate_role(role)
    if not ok:
        logger.warning("create_user: invalid role — %s", err)
        return False

    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()

        # Check duplicate
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            logger.warning("create_user: username already exists — %s", username)
            conn.close()
            return False

        now = now_str()
        password_hash = _hash_password(password)
        role = role.lower()

        try:
            cursor.execute("""
                INSERT INTO users
                (username, password_hash, role, full_name, email, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, password_hash, role, full_name, email, now, now))
            conn.commit()
            logger.info("Created user '%s' with role '%s'", username, role)
            return True
        except Exception as exc:
            logger.error("create_user failed: %s", exc)
            return False
        finally:
            conn.close()


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve user by username.
    Returns None if not found.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def update_user_role(username: str, new_role: str) -> bool:
    """
    Update a user's role.
    """
    ok, err = validate_role(new_role)
    if not ok:
        logger.warning("update_user_role: %s", err)
        return False

    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        now = now_str()
        cursor.execute("""
            UPDATE users SET role = ?, updated_at = ?
            WHERE username = ?
        """, (new_role.lower(), now, username))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        if success:
            logger.info("Updated user '%s' role to '%s'", username, new_role)
        return success


def delete_user(username: str) -> bool:
    """
    Delete a user account.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        if success:
            logger.info("Deleted user '%s'", username)
        return success


def list_users() -> List[Dict[str, Any]]:
    """
    List all users (excluding password hash).
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT username, role, full_name, email, failed_attempts,
               locked_until, created_at, last_login
        FROM users
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ─────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────

def _is_locked(locked_until: Optional[str]) -> bool:
    """Check if account is currently locked."""
    if not locked_until:
        return False
    try:
        lock_time = datetime.fromisoformat(locked_until)
        return datetime.now() < lock_time
    except ValueError:
        return False


def _update_failed_attempts(username: str) -> None:
    """Increment failed attempts and lock account if threshold exceeded."""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        now = now_str()

        cursor.execute("""
            UPDATE users
            SET failed_attempts = failed_attempts + 1,
                updated_at = ?
            WHERE username = ?
        """, (now, username))

        # Check if we need to lock
        cursor.execute(
            "SELECT failed_attempts FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row and row["failed_attempts"] >= MAX_LOGIN_ATTEMPTS:
            lock_until = (
                datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MIN)
            ).isoformat()
            cursor.execute("""
                UPDATE users SET locked_until = ? WHERE username = ?
            """, (lock_until, username))
            logger.warning(
                "Account '%s' locked until %s after %d failed attempts",
                username, lock_until, row["failed_attempts"],
            )

        conn.commit()
        conn.close()


def _reset_failed_attempts(username: str) -> None:
    """Reset failed attempts on successful login."""
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        now = now_str()
        cursor.execute("""
            UPDATE users
            SET failed_attempts = 0, locked_until = NULL, last_login = ?, updated_at = ?
            WHERE username = ?
        """, (now, now, username))
        conn.commit()
        conn.close()


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate user credentials.

    Returns
    -------
    user dict on success, None on failure.
    Implements account lockout policy.
    """
    # Validate inputs
    ok, err = validate_username(username)
    if not ok:
        logger.warning("authenticate: invalid username format")
        return None

    if not password:
        logger.warning("authenticate: empty password")
        return None

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        logger.warning("authenticate: user not found — %s", username)
        return None

    user = dict(row)

    # Check lockout
    if _is_locked(user.get("locked_until")):
        logger.warning("authenticate: account locked — %s", username)
        return None

    # Check max attempts before verifying password
    ok, err = validate_login_attempt(user.get("failed_attempts", 0))
    if not ok:
        logger.warning("authenticate: max attempts exceeded — %s", username)
        return None

    # Verify password
    if not _verify_password(password, user["password_hash"]):
        logger.warning("authenticate: invalid password — %s", username)
        _update_failed_attempts(username)
        return None

    # Success
    _reset_failed_attempts(username)
    logger.info("authenticate: success — %s", username)

    # Remove sensitive fields
    user.pop("password_hash", None)
    return user


# ─────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────

def create_session(user: Dict[str, Any]) -> str:
    """
    Create a new session for an authenticated user.
    Returns session token.
    """
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    expires = now + timedelta(hours=SESSION_TTL_HOURS)

    session_data = {
        "token":      token,
        "username":   user["username"],
        "role":       user["role"],
        "full_name":  user.get("full_name", user["username"]),
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
    }

    # Store in memory
    with _sessions_lock:
        _sessions[token] = session_data

    # Store in database (persistent)
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (token, username, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (token, user["username"], session_data["created_at"], session_data["expires_at"]))
        conn.commit()
        conn.close()

    logger.info(
        "Created session for '%s' — expires in %d hours",
        user["username"], SESSION_TTL_HOURS,
    )
    return token


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a session token.
    Returns session data if valid, None otherwise.
    """
    if not token:
        return None

    # Check memory first
    with _sessions_lock:
        session = _sessions.get(token)
        if session:
            expires = datetime.fromisoformat(session["expires_at"])
            if datetime.now() < expires:
                return session
            else:
                # Expired — remove from memory
                del _sessions[token]

    # Check database
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, u.role, u.full_name
        FROM sessions s
        JOIN users u ON s.username = u.username
        WHERE s.token = ? AND s.is_active = 1
    """, (token,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    session = dict(row)
    expires = datetime.fromisoformat(session["expires_at"])
    if datetime.now() >= expires:
        # Expired — deactivate
        with _db_lock:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET is_active = 0 WHERE token = ?",
                (token,),
            )
            conn.commit()
            conn.close()
        return None

    # Cache in memory
    with _sessions_lock:
        _sessions[token] = {
            "token":      token,
            "username":   session["username"],
            "role":       session["role"],
            "full_name":  session["full_name"],
            "created_at": session["created_at"],
            "expires_at": session["expires_at"],
        }

    return _sessions[token]


def clear_session(token: str) -> None:
    """
    Invalidate a session token.
    """
    with _sessions_lock:
        _sessions.pop(token, None)

    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET is_active = 0 WHERE token = ?",
            (token,),
        )
        conn.commit()
        conn.close()

    logger.info("Cleared session token=%s", mask_key(token, visible=4))


def cleanup_expired_sessions() -> int:
    """
    Remove expired sessions from database.
    Returns count of cleaned sessions.
    """
    now = now_str()
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM sessions WHERE expires_at < ? OR is_active = 0",
            (now,),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()

    # Clean memory
    with _sessions_lock:
        expired = [
            t for t, s in _sessions.items()
            if datetime.fromisoformat(s["expires_at"]) < datetime.now()
        ]
        for t in expired:
            del _sessions[t]

    if count > 0:
        logger.info("Cleaned up %d expired sessions", count)
    return count


# ─────────────────────────────────────────────
# Permission checks
# ─────────────────────────────────────────────

def check_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.
    Wrapper around ROLE_PERMISSIONS from constants.
    """
    if not role or not permission:
        return False
    perms = ROLE_PERMISSIONS.get(role.lower(), [])
    return permission in perms


def user_has_permission(user: Optional[Dict[str, Any]], permission: str) -> bool:
    """
    Check if a user object has a specific permission.
    """
    if not user:
        return False
    role = user.get("role", "")
    return check_permission(role, permission)


# ─────────────────────────────────────────────
# Default admin creation
# ─────────────────────────────────────────────

def create_default_admin() -> bool:
    """
    Create a default admin user if no users exist.
    Uses environment variables or secure defaults.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    row = cursor.fetchone()
    conn.close()

    if row and row["cnt"] > 0:
        return False  # Users already exist

    # Get credentials from env or use defaults
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    # admin_pass = os.getenv("ADMIN_PASSWORD", "")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin")

    if not admin_pass:
        # Generate secure random password
        admin_pass = secrets.token_urlsafe(16)
        logger.warning(
            "ADMIN_PASSWORD not set — generated secure password: %s",
            admin_pass,
        )
        logger.warning("⚠️  Save this password securely! It will not be shown again.")

    success = create_user(
        username=admin_user,
        password=admin_pass,
        role=ROLE_ADMIN,
        full_name="System Administrator",
    )

    if success:
        logger.info("Default admin user created: %s", admin_user)
    else:
        logger.error("Failed to create default admin user")

    return success


# ─────────────────────────────────────────────
# Streamlit integration helpers
# ─────────────────────────────────────────────

def login(username: str, password: str) -> Optional[str]:
    """
    Convenience function for Streamlit login flow.
    Returns session token on success, None on failure.
    """
    user = authenticate(username, password)
    if user:
        return create_session(user)
    return None


def logout(token: str) -> None:
    """
    Convenience function for Streamlit logout flow.
    """
    clear_session(token)


def get_session_user(token: str) -> Optional[Dict[str, Any]]:
    """
    Get user data from session token.
    Returns None if session is invalid.
    """
    session = validate_session(token)
    if session:
        return {
            "username":  session["username"],
            "role":      session["role"],
            "full_name": session["full_name"],
        }
    return None


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────

try:
    init_auth_db()
    create_default_admin()
except Exception as _exc:
    logger.error("Auth service initialization failed: %s", _exc)