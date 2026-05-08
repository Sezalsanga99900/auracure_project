# =============================================================================
# tests/test_auth.py
# AuraEcho+ — Authentication Service Tests
#
# Coverage:
#     • Password hashing (bcrypt + PBKDF2 fallback)
#     • User CRUD operations (create, get, update, delete, list)
#     • Authentication (success, failure, lockout, max attempts)
#     • Session management (create, validate, clear, cleanup, expiry)
#     • Permission checks (role-based access)
#     • Default admin creation
#     • Streamlit helpers (login, logout, get_session_user)
#     • Validation integration
#
# Run:
#     pytest tests/test_auth.py -v
# =============================================================================

import pytest
import os
import time
import sqlite3
import tempfile
import secrets
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.auth_service import (
    # Password hashing
    _hash_password,
    _verify_password,
    # DB
    init_auth_db,
    _get_connection,
    # User operations
    create_user,
    get_user,
    update_user_role,
    delete_user,
    list_users,
    # Auth
    authenticate,
    _is_locked,
    _update_failed_attempts,
    _reset_failed_attempts,
    # Session
    create_session,
    validate_session,
    clear_session,
    cleanup_expired_sessions,
    _sessions,
    _sessions_lock,
    # Permissions
    check_permission,
    user_has_permission,
    # Default admin
    create_default_admin,
    # Streamlit helpers
    login,
    logout,
    get_session_user,
)
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
)
from utils.validators import (
    validate_username,
    validate_password,
    validate_role,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_auth_db(monkeypatch):
    """
    Create a temporary auth database for testing.
    Isolates tests from the real database.
    """
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    # Patch AUTH_DB_PATH
    monkeypatch.setattr("services.auth_service.AUTH_DB_PATH", temp_path)
    
    # Initialize DB
    init_auth_db()
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def clear_sessions():
    """Clear in-memory sessions before each test."""
    with _sessions_lock:
        _sessions.clear()
    yield
    with _sessions_lock:
        _sessions.clear()


@pytest.fixture
def sample_user(temp_auth_db):
    """Create a sample user for testing."""
    username = "testuser"
    password = "SecurePass123!"
    role = ROLE_DOCTOR
    
    success = create_user(
        username=username,
        password=password,
        role=role,
        full_name="Test User",
        email="test@example.com",
    )
    assert success
    
    return {
        "username": username,
        "password": password,
        "role": role,
    }


@pytest.fixture
def sample_session(sample_user, clear_sessions):
    """Create a sample session for testing."""
    user = get_user(sample_user["username"])
    token = create_session(user)
    return {
        "token": token,
        "user": user,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Password Hashing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify_bcrypt(self):
        """Test bcrypt hashing and verification."""
        password = "SecurePass123!"
        hashed = _hash_password(password)
        
        # Hash should start with bcrypt identifier
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        
        # Verification should succeed
        assert _verify_password(password, hashed) is True
        
        # Wrong password should fail
        assert _verify_password("WrongPass123!", hashed) is False

    def test_hash_and_verify_pbkdf2_fallback(self, monkeypatch):
        """Test PBKDF2 fallback when bcrypt unavailable."""
        # Mock bcrypt as unavailable
        monkeypatch.setattr("services.auth_service.BCRYPT_AVAILABLE", False)
        
        password = "SecurePass123!"
        hashed = _hash_password(password)
        
        # Hash should be PBKDF2 format
        assert hashed.startswith("pbkdf2$")
        
        # Verification should succeed
        assert _verify_password(password, hashed) is True
        
        # Wrong password should fail
        assert _verify_password("WrongPass123!", hashed) is False

    def test_verify_bcrypt_with_fallback_disabled(self, monkeypatch):
        """Test bcrypt verification fails gracefully if bcrypt not installed."""
        password = "SecurePass123!"
        hashed = _hash_password(password)  # Creates bcrypt hash
        
        # Mock bcrypt as unavailable
        monkeypatch.setattr("services.auth_service.BCRYPT_AVAILABLE", False)
        
        # Should fail (can't verify bcrypt without bcrypt)
        assert _verify_password(password, hashed) is False

    def test_password_hashing_deterministic(self):
        """Test same password produces different hashes (salt)."""
        password = "SecurePass123!"
        hash1 = _hash_password(password)
        hash2 = _hash_password(password)
        
        assert hash1 != hash2  # Salts should differ
        assert _verify_password(password, hash1)
        assert _verify_password(password, hash2)


# ─────────────────────────────────────────────────────────────────────────────
# User Operations Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUserOperations:
    def test_create_user_success(self, temp_auth_db):
        """Test successful user creation."""
        success = create_user(
            username="newuser",
            password="NewPass123!",
            role=ROLE_NURSE,
            full_name="New User",
            email="new@example.com",
        )
        
        assert success is True
        
        # Verify in DB
        user = get_user("newuser")
        assert user is not None
        assert user["username"] == "newuser"
        assert user["role"] == ROLE_NURSE
        assert user["full_name"] == "New User"
        assert user["email"] == "new@example.com"
        assert user["failed_attempts"] == 0
        assert user["locked_until"] is None

    def test_create_user_duplicate(self, sample_user, temp_auth_db):
        """Test duplicate username fails."""
        success = create_user(
            username=sample_user["username"],
            password="AnotherPass123!",
            role=ROLE_NURSE,
        )
        
        assert success is False

    def test_create_user_invalid_username(self, temp_auth_db):
        """Test invalid username rejected."""
        success = create_user(
            username="ab",  # Too short
            password="ValidPass123!",
            role=ROLE_DOCTOR,
        )
        assert success is False

    def test_create_user_invalid_password(self, temp_auth_db):
        """Test invalid password rejected."""
        success = create_user(
            username="validuser",
            password="weak",  # Too short, no special char
            role=ROLE_DOCTOR,
        )
        assert success is False

    def test_create_user_invalid_role(self, temp_auth_db):
        """Test invalid role rejected."""
        success = create_user(
            username="validuser",
            password="ValidPass123!",
            role="superuser",  # Invalid role
        )
        assert success is False

    def test_get_user_not_found(self, temp_auth_db):
        """Test get_user returns None for non-existent user."""
        user = get_user("nonexistent")
        assert user is None

    def test_update_user_role(self, sample_user, temp_auth_db):
        """Test updating user role."""
        success = update_user_role(sample_user["username"], ROLE_ADMIN)
        assert success is True
        
        user = get_user(sample_user["username"])
        assert user["role"] == ROLE_ADMIN

    def test_update_user_role_invalid(self, sample_user, temp_auth_db):
        """Test updating to invalid role fails."""
        success = update_user_role(sample_user["username"], "invalid_role")
        assert success is False

    def test_delete_user(self, sample_user, temp_auth_db):
        """Test deleting user."""
        success = delete_user(sample_user["username"])
        assert success is True
        
        user = get_user(sample_user["username"])
        assert user is None

    def test_delete_user_not_found(self, temp_auth_db):
        """Test deleting non-existent user returns False."""
        success = delete_user("nonexistent")
        assert success is False

    def test_list_users(self, temp_auth_db):
        """Test listing users."""
        create_user("user1", "Pass123!a", ROLE_DOCTOR)
        create_user("user2", "Pass123!b", ROLE_NURSE)
        
        users = list_users()
        
        assert len(users) == 2
        usernames = [u["username"] for u in users]
        assert "user1" in usernames
        assert "user2" in usernames
        
        # Password hash should not be in list
        for user in users:
            assert "password_hash" not in user


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthentication:
    def test_authenticate_success(self, sample_user, temp_auth_db):
        """Test successful authentication."""
        user = authenticate(sample_user["username"], sample_user["password"])
        
        assert user is not None
        assert user["username"] == sample_user["username"]
        assert "password_hash" not in user  # Sensitive field removed

    def test_authenticate_user_not_found(self, temp_auth_db):
        """Test authentication fails for non-existent user."""
        user = authenticate("nonexistent", "Password123!")
        assert user is None

    def test_authenticate_wrong_password(self, sample_user, temp_auth_db):
        """Test authentication fails for wrong password."""
        user = authenticate(sample_user["username"], "WrongPass123!")
        assert user is None

    def test_authenticate_empty_password(self, sample_user, temp_auth_db):
        """Test authentication fails for empty password."""
        user = authenticate(sample_user["username"], "")
        assert user is None

    def test_account_lockout_after_max_attempts(self, sample_user, temp_auth_db):
        """
        Test account locks after MAX_LOGIN_ATTEMPTS failed attempts.
        """
        # Make MAX_LOGIN_ATTEMPTS failed attempts
        for i in range(MAX_LOGIN_ATTEMPTS):
            user = authenticate(sample_user["username"], "WrongPass123!")
            assert user is None
        
        # Check user is locked
        user = get_user(sample_user["username"])
        assert user["failed_attempts"] == MAX_LOGIN_ATTEMPTS
        assert user["locked_until"] is not None
        
        # Next attempt should fail even with correct password
        user = authenticate(sample_user["username"], sample_user["password"])
        assert user is None

    def test_account_lockout_duration(self, sample_user, temp_auth_db, monkeypatch):
        """Test account unlocks after LOCKOUT_DURATION_MIN."""
        # Lock the account
        for _ in range(MAX_LOGIN_ATTEMPTS):
            authenticate(sample_user["username"], "WrongPass123!")
        
        user = get_user(sample_user["username"])
        assert _is_locked(user["locked_until"]) is True
        
        # Mock time to be after lockout duration
        lock_time = datetime.fromisoformat(user["locked_until"])
        future_time = lock_time + timedelta(minutes=LOCKOUT_DURATION_MIN + 1)
        
        monkeypatch.setattr("services.auth_service.datetime", MagicMock(now=lambda: future_time))
        
        # Should be unlocked now
        assert _is_locked(user["locked_until"]) is False
        
        # Authentication should succeed
        user = authenticate(sample_user["username"], sample_user["password"])
        assert user is not None

    def test_failed_attempts_reset_on_success(self, sample_user, temp_auth_db):
        """Test failed attempts reset on successful login."""
        # Make some failed attempts
        authenticate(sample_user["username"], "WrongPass123!")
        authenticate(sample_user["username"], "WrongPass123!")
        
        user = get_user(sample_user["username"])
        assert user["failed_attempts"] == 2
        
        # Successful login
        user = authenticate(sample_user["username"], sample_user["password"])
        assert user is not None
        
        # Failed attempts should be reset
        user = get_user(sample_user["username"])
        assert user["failed_attempts"] == 0
        assert user["locked_until"] is None
        assert user["last_login"] is not None

    def test_validate_login_attempt_helper(self):
        """Test validate_login_attempt from validators."""
        from utils.validators import validate_login_attempt
        
        ok, msg = validate_login_attempt(3)
        assert ok is True
        
        ok, msg = validate_login_attempt(MAX_LOGIN_ATTEMPTS)
        assert ok is False
        assert "locked" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Session Management Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionManagement:
    def test_create_session(self, sample_user, clear_sessions, temp_auth_db):
        """Test session creation."""
        user = get_user(sample_user["username"])
        token = create_session(user)
        
        assert token is not None
        assert len(token) > 20  # Secure token
        
        # Check in memory
        assert token in _sessions
        assert _sessions[token]["username"] == user["username"]
        
        # Check in DB
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row["username"] == user["username"]

    def test_validate_session_valid(self, sample_session, temp_auth_db):
        """Test validating a valid session."""
        session = validate_session(sample_session["token"])
        
        assert session is not None
        assert session["username"] == sample_session["user"]["username"]
        assert session["role"] == sample_session["user"]["role"]

    def test_validate_session_invalid_token(self, clear_sessions, temp_auth_db):
        """Test validating invalid token returns None."""
        session = validate_session("invalid_token")
        assert session is None

    def test_validate_session_expired(self, sample_session, temp_auth_db, monkeypatch):
        """Test expired session returns None."""
        # Mock time to be after expiry
        expiry = datetime.fromisoformat(_sessions[sample_session["token"]]["expires_at"])
        future_time = expiry + timedelta(hours=1)
        
        monkeypatch.setattr("services.auth_service.datetime", MagicMock(now=lambda: future_time))
        
        session = validate_session(sample_session["token"])
        assert session is None
        
        # Session should be removed from memory
        assert sample_session["token"] not in _sessions

    def test_clear_session(self, sample_session, temp_auth_db):
        """Test clearing a session."""
        clear_session(sample_session["token"])
        
        # Removed from memory
        assert sample_session["token"] not in _sessions
        
        # Deactivated in DB
        session = validate_session(sample_session["token"])
        assert session is None

    def test_cleanup_expired_sessions(self, sample_user, clear_sessions, temp_auth_db, monkeypatch):
        """Test cleanup removes expired sessions."""
        # Create multiple sessions
        user = get_user(sample_user["username"])
        token1 = create_session(user)
        token2 = create_session(user)
        
        # Expire token1
        expiry1 = datetime.fromisoformat(_sessions[token1]["expires_at"])
        future_time = expiry1 + timedelta(hours=1)
        
        # Manually set token1 as expired in memory for test
        _sessions[token1]["expires_at"] = (datetime.now() - timedelta(hours=1)).isoformat()
        
        # Run cleanup
        count = cleanup_expired_sessions()
        
        # token1 should be removed
        assert token1 not in _sessions
        assert token2 in _sessions  # token2 still valid

    def test_session_ttl_hours(self, sample_user, clear_sessions, temp_auth_db):
        """Test session expires after SESSION_TTL_HOURS."""
        user = get_user(sample_user["username"])
        token = create_session(user)
        
        session = _sessions[token]
        created = datetime.fromisoformat(session["created_at"])
        expires = datetime.fromisoformat(session["expires_at"])
        
        delta = expires - created
        assert delta.total_seconds() == SESSION_TTL_HOURS * 3600


# ─────────────────────────────────────────────────────────────────────────────
# Permission Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissions:
    def test_check_permission_valid(self):
        """Test permission check for valid role/permission."""
        assert check_permission(ROLE_DOCTOR, "view_dashboard") is True
        assert check_permission(ROLE_DOCTOR, "edit_patient") is True
        assert check_permission(ROLE_NURSE, "enter_vitals") is True

    def test_check_permission_invalid(self):
        """Test permission check for invalid role/permission."""
        assert check_permission(ROLE_NURSE, "edit_patient") is False
        assert check_permission(ROLE_DOCTOR, "manage_users") is False
        assert check_permission("invalid_role", "view_dashboard") is False

    def test_check_permission_empty_inputs(self):
        """Test permission check handles empty inputs."""
        assert check_permission("", "view_dashboard") is False
        assert check_permission(ROLE_DOCTOR, "") is False
        assert check_permission(None, "view_dashboard") is False

    def test_user_has_permission(self):
        """Test user_has_permission helper."""
        user = {"role": ROLE_DOCTOR}
        assert user_has_permission(user, "view_dashboard") is True
        assert user_has_permission(user, "manage_users") is False
        
        # None user
        assert user_has_permission(None, "view_dashboard") is False


# ─────────────────────────────────────────────────────────────────────────────
# Default Admin Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultAdmin:
    def test_create_default_admin_when_empty(self, temp_auth_db, monkeypatch):
        """Test default admin created when no users exist."""
        # Set admin credentials via env
        monkeypatch.setenv("ADMIN_USERNAME", "testadmin")
        monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
        
        result = create_default_admin()
        
        assert result is True
        
        user = get_user("testadmin")
        assert user is not None
        assert user["role"] == ROLE_ADMIN

    def test_create_default_admin_skips_if_users_exist(self, sample_user, temp_auth_db):
        """Test default admin not created if users already exist."""
        result = create_default_admin()
        assert result is False

    def test_create_default_admin_generates_password(self, temp_auth_db, monkeypatch, caplog):
        """Test secure password generated if ADMIN_PASSWORD not set."""
        monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
        monkeypatch.setenv("ADMIN_USERNAME", "genadmin")
        
        result = create_default_admin()
        
        assert result is True
        
        # Check log for generated password warning
        assert "generated secure password" in caplog.text.lower()
        
        # User should exist
        user = get_user("genadmin")
        assert user is not None


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Helpers Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamlitHelpers:
    def test_login_success(self, sample_user, clear_sessions, temp_auth_db):
        """Test login returns token on success."""
        token = login(sample_user["username"], sample_user["password"])
        
        assert token is not None
        assert token in _sessions

    def test_login_failure(self, sample_user, clear_sessions, temp_auth_db):
        """Test login returns None on failure."""
        token = login(sample_user["username"], "WrongPass123!")
        assert token is None

    def test_logout(self, sample_session, temp_auth_db):
        """Test logout clears session."""
        logout(sample_session["token"])
        
        assert sample_session["token"] not in _sessions
        assert validate_session(sample_session["token"]) is None

    def test_get_session_user_valid(self, sample_session, temp_auth_db):
        """Test get_session_user returns user data."""
        user = get_session_user(sample_session["token"])
        
        assert user is not None
        assert user["username"] == sample_session["user"]["username"]
        assert user["role"] == sample_session["user"]["role"]
        assert "full_name" in user

    def test_get_session_user_invalid(self, clear_sessions, temp_auth_db):
        """Test get_session_user returns None for invalid token."""
        user = get_session_user("invalid_token")
        assert user is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_auth_flow(self, temp_auth_db, clear_sessions):
        """Test complete authentication flow."""
        # 1. Create user
        success = create_user(
            username="flowuser",
            password="FlowPass123!",
            role=ROLE_DOCTOR,
        )
        assert success
        
        # 2. Login
        token = login("flowuser", "FlowPass123!")
        assert token is not None
        
        # 3. Validate session
        session = validate_session(token)
        assert session is not None
        assert session["username"] == "flowuser"
        assert session["role"] == ROLE_DOCTOR
        
        # 4. Check permissions
        assert check_permission(session["role"], "edit_patient") is True
        assert check_permission(session["role"], "manage_users") is False
        
        # 5. Logout
        logout(token)
        assert validate_session(token) is None

    def test_lockout_prevents_session_creation(self, sample_user, temp_auth_db, clear_sessions):
        """Test locked account cannot create session."""
        # Lock account
        for _ in range(MAX_LOGIN_ATTEMPTS):
            authenticate(sample_user["username"], "WrongPass123!")
        
        # Try to login
        token = login(sample_user["username"], sample_user["password"])
        assert token is None
        
        # No session created
        assert len(_sessions) == 0