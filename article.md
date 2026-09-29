# When Hindsight failed, I still had the facts

The first version of this system had a dangerous failure mode: when answer generation failed, the user got an error-shaped experience even though retrieval had already found useful organizational evidence. I did not want an unavailable model to erase available facts. So I made a stricter promise: if Hindsight can recall relevant information, the application should still return that information, even when the final language-generation step is unavailable.

That decision shaped the rest of the project. I built a FastAPI backend, a React frontend, and a memory layer around [Hindsight GitHub](https://github.com/vectorize-io/hindsight). The system ingests current organizational references, stores experience records with provenance, recalls relevant memories, and uses an LLM to turn those memories into an answer. The important part is the boundary between recall and generation. They are separate operations, and the application treats them differently when something goes wrong.

## The system has two jobs

The repository is an organizational knowledge system with two related jobs.

The first job is retention. When I upload a reference document or record an outcome, the backend assigns it an identity, attaches metadata, and stores it in a Hindsight memory bank. The record can include a title, document type, project, department, entities, tags, timestamps, and version information.

The second job is answering. When someone asks a question, the backend searches Hindsight for current organizational facts and user feedback, builds a bounded context, and asks the model to answer only from that context. The frontend exposes the result through the Memory Lab, including confidence, retrieved count, and sources.

That separation sounds obvious, but it is easy to lose in an AI application. Retrieval, synthesis, and presentation often collapse into one request. When that happens, an outage in one layer becomes an outage in every layer. I wanted the system to preserve the strongest result available at each stage:

- no recall means an explicit insufficient-information answer;
- recall without generation means retrieved facts with a lower confidence state;
- successful recall and generation means a grounded answer with sources.

That is the core design story. The model is useful, but it is not the only useful component.

## Hindsight is the evidence layer

The backend wraps Hindsight in a small asynchronous client. The wrapper keeps application code from spreading URL construction, authorization headers, request timeouts, and response parsing across the rest of the project.

The client also preserves the metadata that makes retrieval reviewable:

```python
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
```

This is more than a convenience wrapper. It establishes what the application means by recall. The query is explicit, the result limit is bounded, and tags can narrow the search to the appropriate source. Hindsight is acting as an operational memory bank rather than an invisible prompt cache.

The application stores records with the same care. A document is assigned a stable identifier, and its content hash and metadata are checked before another copy is retained. If the content and identifying metadata are unchanged, ingestion returns an `unchanged` status. If the document has changed, the service increments its version before storing the new record.

That matters during a failure. If the system has to return raw retrieved facts, those facts need to carry enough context for a person to judge them. A string without a title, type, or source is not a useful fallback; it is just an unexplained fragment.

## The contract: never throw away successful retrieval

The answering path in `HindsightRAG` is intentionally staged. It first recalls memories. If there are no results, it returns a clear insufficient-information response. If there are results, it builds a source-labeled context and attempts generation.

The empty-result branch is deliberately conservative:

```python
if not search_results:
    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "confidence": "insufficient",
        "retrieved_count": 0,
    }
```

This is not a minor UX detail. It prevents the model from inventing an answer when the organization’s memory bank has no supporting material. An absence of evidence is represented as a state the client can understand, not hidden behind a generic response.

When retrieval succeeds, the service formats each memory with a source number, title, type, and content. The language model receives that context along with an instruction to cite the source title and admit when the context does not support the answer.

The final boundary is the most important one:

```python
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
```

The exception handler does not pretend that a generated answer exists. It returns the evidence directly and labels the result `retrieved_only`. That label is important: it tells the frontend and the operator that retrieval worked but synthesis did not.

This is the kind of graceful degradation I want in a system that people may use during an incident. The fallback is not equally polished, and it should not be presented as if it were. But it is materially better than hiding the evidence because one downstream dependency failed.

## A concrete example: an HTTP 429 question

Consider the webhook case represented in the repository. A payment partner returns HTTP 429 during a traffic spike. Previous fixed-interval retries synchronized and extended the queue backlog. A later outcome recommends capped backoff, jitter, bounded attempts, and idempotent processing.

With a healthy generation path, a question about the incident can produce a concise recommendation grounded in those records. The answer can explain that synchronized retries made the failure worse and point to the earlier outcome as evidence.

If the model call fails, the user still receives the recalled records. They may see the source title, the stored outcome, the retry lesson, and the associated metadata instead of a synthesized paragraph. That is not the same interaction, but it is still actionable. An engineer can inspect the evidence and make a decision without waiting for the generation service to recover.

If Hindsight returns no relevant records, the system does something different again: it says that the organizational knowledge base does not contain enough information to answer confidently. That distinction prevents two common errors. It avoids hallucinating when recall is empty, and it avoids discarding useful evidence when generation is unavailable.

The frontend makes these states visible. The Memory Lab can show the baseline and memory-informed reasoning, the retrieved memories, source details, and the confidence state. That visibility is essential because the user should know whether they are reading a generated answer, a retrieval-only result, or an explicit information gap.

## Why provenance matters during failure

Fallback behavior only works if the memory records are trustworthy enough to inspect. This is why the project keeps current references and historical experiences distinct.

Current reference records describe the organization as it is supposed to operate: ownership, documents, policies, and other present facts. Experience records describe what happened in a prior situation, including the action taken, outcome status, and lesson. A lesson is useful evidence, but it is not automatically current policy.

The repository carries that distinction through tags and metadata. Records can identify their source, data type, document version, scenario, project, and reliability. Synthetic organizational records are explicitly marked as such, so they cannot quietly masquerade as verified operational history. In a completed production implementation, the same fields can represent authenticated documents, incident records, change reviews, and user-attributed outcomes.

This is also where [Vectorize agent memory](https://vectorize.io/what-is-agent-memory) is a useful framing: memory is an operational capability, not just additional text in a prompt. The application needs to know what it remembered, where it came from, and how much confidence to assign when turning it into a recommendation.

The [Hindsight docs](https://hindsight.vectorize.io/) describe the underlying memory service, but the application-level benefit is straightforward. I can store structured evidence, recall it with tags and query context, and expose the retrieved records when synthesis is unavailable. That gives the system a useful failure boundary instead of one large all-or-nothing request.

## What I learned building the failure path first

The first lesson was that retrieval and generation should have separate contracts. Retrieval answers, “what relevant evidence exists?” Generation answers, “how should that evidence be explained?” Those questions can fail independently, so the code should represent them independently.

The second lesson was that confidence needs to describe the path taken. `grounded`, `retrieved_only`, and `insufficient` are more useful than a single success flag. They let the frontend communicate whether the system generated an answer, returned evidence directly, or found nothing relevant.

The third lesson was that source metadata is part of reliability, not decoration. A fallback response is only useful when a person can tell what each fact is, where it came from, and whether it is current, historical, or synthetic.

The fourth lesson was that conservative failure is better than confident completion. When recall is empty, the system should not improvise an organizational answer. When generation fails, it should not hide retrieved facts. Those are different failures and deserve different responses.

Finally, I learned that resilience can make an AI system more honest. The fallback does not try to imitate the model. It says exactly what happened: the system found facts, but answer generation is unavailable. That is a better operational contract than a polished response whose provenance is unclear.

That is what I would keep from this project. Hindsight provides the persistent memory layer, but the application decides how to use that memory responsibly. It separates current references from historical outcomes, labels provenance, compares retrieval with generation, and preserves evidence when the model path breaks. When Hindsight failed, I still had the facts. That is the behavior I want from any system people depend on when the convenient path is unavailable.
