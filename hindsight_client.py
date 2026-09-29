from typing import Any

import httpx

from app.config import settings


class HindsightClient:
    """Small async client for the Hindsight Cloud REST API."""

    def __init__(self):
        self.base_url = settings.HINDSIGHT_BASE_URL.rstrip("/")
        self.api_key = settings.HINDSIGHT_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _bank_path(self, bank_id: str) -> str:
        if not bank_id:
            raise RuntimeError("HINDSIGHT_BANK_ID is not configured")
        return f"/v1/default/banks/{bank_id}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method, f"{self.base_url}{path}", headers=self.headers, **kwargs
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    async def store_knowledge(self, bank_id: str, content: str,
                              metadata: dict[str, Any]) -> dict[str, Any]:
        """Retain content and its searchable metadata in a memory bank."""
        item = {
            "content": content,
            "metadata": {key: str(value) for key, value in metadata.items()
                         if value is not None and key not in {"tags", "entities"}},
            "context": metadata.get("doc_type", "GENERAL"),
        }
        if metadata.get("document_id"):
            item["document_id"] = str(metadata["document_id"])
        if metadata.get("occurred_at"):
            item["timestamp"] = str(metadata["occurred_at"])
        if metadata.get("entities"):
            item["entities"] = [
                entity if isinstance(entity, dict) else {"text": str(entity), "type": "CONCEPT"}
                for entity in metadata["entities"]
            ]
        if metadata.get("tags"):
            item["tags"] = [str(tag) for tag in metadata["tags"]]
        return await self._request(
            "POST", f"{self._bank_path(bank_id)}/memories", json={"items": [item]}
        )

    async def search_memories(self, bank_id: str, query: str,
                              limit: int = 8, tags: list[str] | None = None,
                              tags_match: str = "any",
                              prefer_observations: bool = True) -> list[dict[str, Any]]:
        """Recall facts from a memory bank."""
        payload: dict[str, Any] = {
            "query": query,
            "max_tokens": max(256, limit * 512),
            "prefer_observations": prefer_observations,
        }
        if tags is not None:
            payload["tags"] = tags
            payload["tags_match"] = tags_match
        response = await self._request(
            "POST",
            f"{self._bank_path(bank_id)}/memories/recall",
            json=payload,
        )
        return response.get("results", [])

    async def list_memories(self, bank_id: str, limit: int = 100,
                            tags: list[str] | None = None,
                            tags_match: str = "any_strict") -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if tags is not None:
            params["tags"] = tags
            params["tags_match"] = tags_match
        return await self._request(
            "GET", f"{self._bank_path(bank_id)}/memories/list", params=params
        )

    async def list_documents(self, bank_id: str, limit: int = 100,
                             tags: list[str] | None = None,
                             tags_match: str = "any_strict") -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if tags is not None:
            params["tags"] = tags
            params["tags_match"] = tags_match
        return await self._request(
            "GET", f"{self._bank_path(bank_id)}/documents", params=params
        )

    async def get_document(self, bank_id: str, document_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"{self._bank_path(bank_id)}/documents/{document_id}"
        )

    async def get_bank_stats(self, bank_id: str) -> dict[str, Any]:
        return await self._request("GET", f"{self._bank_path(bank_id)}/stats")