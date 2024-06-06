# session_manager.py
import aiohttp
from aiohttp import ClientSession
from typing import Optional


class SessionManager:
    def __init__(self, max_connections: int = 5, hf_token: Optional[str] = None):
        self.max_connections = max_connections
        self.hf_token = hf_token
        self.session: Optional[ClientSession] = None

    async def __aenter__(self) -> "SessionManager":
        self.session = ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_connections),
            headers=(
                {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_session(self) -> ClientSession:
        if self.session is None:
            raise ValueError("Session is not initialized.")
        return self.session
