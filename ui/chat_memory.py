# multi_agent_rag/chat_memory.py

from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class Message:
    sender: str
    role: str  # e.g., "user", "agent", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ChatSession:
    session_id: str
    history: List[Message] = field(default_factory=list)

    def add_message(self, sender: str, role: str, content: str):
        self.history.append(Message(sender, role,content))

    def get_history(self) -> List[Dict]:
        return [msg.__dict__ for msg in self.history]

class ChatMemory:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def get_or_create_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]

    def add_message(self, session_id: str, sender: str, role: str, content: str):
        session = self.get_or_create_session(session_id)
        session.add_message(sender, role, content)

    def get_history(self, session_id: str) -> List[Dict]:
        session = self.sessions.get(session_id)
        return session.get_history() if session else []
