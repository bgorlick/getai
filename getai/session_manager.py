"""session_manager.py for GetAI - Contains the SessionManager class for managing aiohttp.ClientSession instances."""

from typing import Optional
import aiohttp
from aiohttp import ClientSession


class SessionManager:
    """Manages aiohttp.ClientSession with optional auth and connection limit."""

    def __init__(self, max_connections: int = 5, hf_token: Optional[str] = None):
        """Initializes SessionManager with max connections and optional token."""
        self.max_connections = max_connections
        self.hf_token = hf_token
        self.session: Optional[ClientSession] = None

    async def __aenter__(self) -> "SessionManager":
        """Creates and returns a new aiohttp session."""
        self.session = ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_connections),
            headers=(
                {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the aiohttp session."""
        if self.session:
            await self.session.close()

    async def get_session(self) -> ClientSession:
        """Returns the current aiohttp session, raises if not initialized."""
        if self.session is None:
            raise ValueError("Session is not initialized.")
        return self.session
