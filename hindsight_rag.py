import asyncio
import hashlib
import logging
from typing import Any

import httpx

from app.config import settings
from app.services.hindsight_client import HindsightClient
from app.services.llm import llm


FALLBACK_ANSWER = (
    "I couldn't find sufficient information in the organization's knowledge base "
    "to answer this confidently."
)
logger = logging.getLogger(__name__)


class HindsightRAG:
    """Question answering pipeline backed by Hindsight Cloud recall."""

    def __init__(self):
        self.hindsight = HindsightClient()
        self.bank_id = settings.HINDSIGHT_BANK_ID

    async def ingest_document(self, content: str,
                              metadata: dict[str, Any]) -> dict[str, Any]:
        title = metadata.get("title", "Untitled")
        source_reference = metadata.get("source_reference", f"Uploaded reference: {title}")
        document_id = str(metadata.get("document_id") or hashlib.sha256(content.encode("utf-8")).hexdigest())
        tags = [str(tag) for tag in metadata.get("tags", [])]
        tags.extend(["source:organization", "data:current-reference"])
        tags = list(dict.fromkeys(tags))
        version = int(metadata.get("version", 1))
        try:
            existing = await self.hindsight.get_document(self.bank_id, document_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            existing = None
        if existing:
            existing_metadata = existing.get("document_metadata") or {}
            if (
                existing.get("content_hash") == hashlib.sha256(content.encode("utf-8")).hexdigest()
                and existing_metadata.get("title") == title
                and existing_metadata.get("source_reference") == source_reference
            ):
                return {"success": True, "status": "unchanged", "document_id": document_id}
            try:
                version = max(version, int(existing_metadata.get("version", "1")) + 1)
            except (TypeError, ValueError):
                version = max(version, 2)
        hindsight_metadata = {
            "title": title,
            "doc_type": metadata.get("doc_type", "GENERAL"),
            "project": metadata.get("project", ""),
            "department": metadata.get("department", ""),
            "author": metadata.get("author", ""),
            "source": "organization_document",
            "data_type": "current_reference",
            "source_reference": source_reference,
            "document_id": document_id,
            "version": str(version),
            "entities": metadata.get("entities", []),
            "tags": list(dict.fromkeys(tags)),
        }
        result = await self.hindsight.store_knowledge(
            bank_id=self.bank_id,
            content=content,
            metadata=hindsight_metadata
        )
        if result.get("success") is False:
            raise RuntimeError("Hindsight did not confirm retaining the reference document")
        return {**result, "status": "updated" if existing else "stored", "document_id": document_id, "version": version}

    async def answer_question(self, question: str) -> dict[str, Any]:
        search_results = await self.hindsight.search_memories(
            bank_id=self.bank_id,
            query=question,
            limit=8,
            tags=["source:organization", "source:user-feedback"],
            tags_match="any_strict",
        )

        if not search_results:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "confidence": "insufficient",
                "retrieved_count": 0,
            }

        context = self._build_context(search_results)
        system_prompt = """You are the Organizational Brain for Nexus Technologies.
        Answer questions using ONLY the provided organizational context.
        Cite the source title when making a claim. If the context does not support
        an answer, say: "I couldn't find sufficient information in the organization's
        knowledge base to answer this confidently."
        
        Context from organizational memory:
        {context}
        """

        try:
            answer = await asyncio.to_thread(
                llm.chat,
                system_prompt.format(context=context),
                question,
            )
            confidence = "grounded"
        except Exception:
            logger.exception("Groq answer generation failed; returning retrieved facts")
            answer = "I found these relevant facts, but answer generation is unavailable:\n\n" + "\n\n".join(
                memory.get("text", "") for memory in search_results
                if memory.get("text")
            )
            confidence = "retrieved_only"
        sources = self._extract_sources(search_results)
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "retrieved_count": len(search_results)
        }

    def _build_context(self, memories: list[dict[str, Any]]) -> str:
        context_parts = []
        for i, memory in enumerate(memories, 1):
            metadata = memory.get("metadata") or {}
            context_parts.append(
                f"Source {i}: {metadata.get('title', 'Unknown')}\n"
                f"Type: {metadata.get('doc_type', memory.get('type', 'GENERAL'))}\n"
                f"Content: {memory.get('text', '')}\n---"
            )
        return "\n".join(context_parts)

    def _extract_sources(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": memory.get("id"),
                "title": (memory.get("metadata") or {}).get("title", "Unknown"),
                "type": memory.get("type", "world"),
                "text": memory.get("text", ""),
            }
            for memory in memories
        ]