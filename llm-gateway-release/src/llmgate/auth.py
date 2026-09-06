"""Bearer-token authentication. Keys are configured as SHA-256 hashes, so the config file
never contains a secret and a leaked file does not leak access."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llmgate.config import AuthSettings


class AuthError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    authenticated: bool


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Authenticator:
    def __init__(self, settings: AuthSettings) -> None:
        self._s = settings

    def authenticate(self, authorization: str | None) -> Principal:
        if not authorization:
            if self._s.required:
                msg = "missing API key"
                raise AuthError(msg)
            return Principal("anonymous", False)
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            msg = "malformed Authorization header"
            raise AuthError(msg)
        name = self._s.api_keys.get(hash_key(token.strip()))
        if name is None:
            msg = "invalid API key"
            raise AuthError(msg)
        return Principal(name, True)
