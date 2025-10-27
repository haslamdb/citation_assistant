#!/usr/bin/env python3
"""
Authentication and Security Module
JWT-based authentication with password hashing
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import json

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = False


class UserInDB(User):
    hashed_password: str


class UserManager:
    """Manage users in JSON file"""

    def __init__(self, users_file: str = None):
        if users_file is None:
            users_file = str(Path(__file__).parent.parent / "configs" / "users.json")

        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with empty dict if file doesn't exist
        if not self.users_file.exists():
            self._save_users({})

    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        with open(self.users_file, 'r') as f:
            return json.load(f)

    def _save_users(self, users: Dict):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        users = self._load_users()
        if username in users:
            return UserInDB(**users[username])
        return None

    def create_user(self, username: str, password: str, email: str = None, full_name: str = None) -> UserInDB:
        """Create new user"""
        users = self._load_users()

        if username in users:
            raise ValueError(f"User {username} already exists")

        hashed_password = pwd_context.hash(password)
        user = {
            "username": username,
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "disabled": False
        }

        users[username] = user
        self._save_users(users)

        return UserInDB(**user)

    def update_password(self, username: str, new_password: str):
        """Update user password"""
        users = self._load_users()

        if username not in users:
            raise ValueError(f"User {username} not found")

        users[username]["hashed_password"] = pwd_context.hash(new_password)
        self._save_users(users)

    def delete_user(self, username: str):
        """Delete user"""
        users = self._load_users()

        if username not in users:
            raise ValueError(f"User {username} not found")

        del users[username]
        self._save_users(users)

    def list_users(self) -> list:
        """List all usernames"""
        users = self._load_users()
        return list(users.keys())


# Initialize user manager
user_manager = UserManager()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user"""
    user = user_manager.get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    user = user_manager.get_user(username=token_data.username)

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (not disabled)"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user
