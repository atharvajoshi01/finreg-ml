"""Append-only audit log for model lifecycle events."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """A single audit log entry."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event: str
    details: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.checksum is None:
            payload = f"{self.timestamp}:{self.event}:{json.dumps(self.details, sort_keys=True)}"
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()[:16]


class AuditLog:
    """Append-only audit log with integrity checksums.

    Every model training, prediction, and evaluation event is recorded
    with a SHA-256 checksum for tamper detection.
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    def log(self, event: str, **details: Any) -> AuditEntry:
        entry = AuditEntry(event=event, details=details)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_json(self, path: Optional[str] = None) -> str:
        data = [e.model_dump() for e in self._entries]
        output = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.model_dump() for e in self._entries]
