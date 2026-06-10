import json
import time
import redis as redis_lib
from celery import shared_task
from django.utils import timezone
from Ai_strategy_engine.logger import logger


def get_redis():
    return redis_lib.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def emit(thread_id, step, message, done=False, error=False, data=None):
    r = get_redis()
    payload = {"step": step, "message": message, "done": done, "error": error}
    if data:
        payload["data"] = data
    key = f"pipeline:{thread_id}"
    r.rpush(key, json.dumps(payload))
    r.expire(key, 3600)
    logger.info(f"[SSE] {thread_id} | {step}: {message}")


@shared_task
def index_document_task(kb_doc_id: str):
    """Index a KnowledgeBase document into Pinecone. Called after upload."""
    import os
    from .models import KnowledgeBase
    from .pinecone_service import PineconeService
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        doc = KnowledgeBase.objects.get(id=kb_doc_id)
        doc.status = "indexing"
        doc.save(update_fields=["status"])

        file_path = doc.file.path
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in (".docx", ".doc"):
            loader = Docx2txtLoader(file_path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise RuntimeError(f"Unsupported file type: {ext}")

        loaded_docs = loader.load()
        if not loaded_docs:
            raise RuntimeError("No content loaded from document")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(loaded_docs)

        # PyPDFLoader sets metadata["page"] as 0-indexed; add 1 for human page numbers.
        # Docx/TextLoader don't set "page", so it defaults to 0.
        chunks_meta = [
            {"text": s.page_content, "page": (s.metadata.get("page") or 0) + 1}
            for s in splits if s.page_content.strip()
        ]
        if not chunks_meta:
            raise RuntimeError("No text content found in document")

        pc = PineconeService()
        count = pc.index_document(str(doc.id), doc.name, chunks_meta)

        doc.status = "indexed"
        doc.chunk_count = count
        doc.error = ""
        doc.save(update_fields=["status", "chunk_count", "error"])
        logger.info(f"[KB] Indexed doc {doc.id} — {count} chunks")

    except Exception as e:
        logger.error(f"[KB] Indexing failed for {kb_doc_id}: {e}", exc_info=True)
        KnowledgeBase.objects.filter(id=kb_doc_id).update(
            status="failed", error=str(e)
        )


@shared_task
def run_pipeline_task(thread_id, input_query, document_ids=None):
    from .chains import ParallelStrategicAnalysis, AggregatedStrategicAnalysis, normalize_agent_outputs
    from .models import Thread, FinalStrategy, AgentResponse
    from .helper import HybridRAGService

    start = time.time()
    document_ids = document_ids or []

    try:
        Thread.objects.filter(thread_id=thread_id).update(
            status='running', objective=input_query
        )
        emit(thread_id, "start", "Pipeline started")

        # ── Per-agent RAG context from Pinecone (if docs selected) ────
        contexts = {}
        kb_citations_map = {}   # agent_name → list of KB citation dicts
        if document_ids:
            emit(thread_id, "doc_loaded", f"{len(document_ids)} document(s) selected")
            rag = HybridRAGService(document_ids=document_ids)
            _agent_queries = {
                "executive_summary":     "problem opportunity solution market gap innovation",
                "market_analysis":       "market size revenue growth cagr trends forecast",
                "competitive_landscape": "competitors pricing features positioning funding",
                "monetization_strategy": "revenue model pricing unit economics monetization",
                "risk_assessment":       "risks challenges regulatory compliance barriers",
                "roadmap":               "timeline phases milestones execution development",
                "weakness_review":       "weaknesses gaps limitations assumptions risks",
            }
            emit(thread_id, "rag_retrieval", "Querying knowledge base for per-agent context...")
            for agent_name, query in _agent_queries.items():
                ctx, cites = rag.retrieve(query, top_k=2)
                contexts[agent_name]      = ctx
                kb_citations_map[agent_name] = cites
            emit(thread_id, "rag_done", "Context retrieved — ready for agents")
        else:
            emit(thread_id, "doc_loaded", "No documents selected — running on objective only")

        # ── AI Agents ──────────────────────────────────────────────
        emit(thread_id, "agents_start", "Launching 7 parallel AI agents...")

        agent_names = [
            "executive_summary", "market_analysis", "competitive_landscape",
            "monetization_strategy", "risk_assessment", "roadmap", "weakness_review",
        ]

        for name in agent_names:
            AgentResponse.objects.get_or_create(
                thread_id=thread_id, agent_name=name,
                defaults={"status": "running"}
            )

        ai_strategy = ParallelStrategicAnalysis(
            objective=input_query, thread_id=thread_id, contexts=contexts
        )
        agents_start_time = time.time()
        strategy_chain_data = ai_strategy.make_parallel_chains().invoke({
            "objective": input_query,
        })
        agents_runtime_ms = int((time.time() - agents_start_time) * 1000)

        # Guarantee every field the frontend reads is present — fill gaps with safe defaults
        strategy_chain_data = normalize_agent_outputs(strategy_chain_data)

        # Inject citations — KB from RAG retrieval + web from tool calls (stored as _web_citations).
        # citations key is guaranteed by _AGENT_DEFAULTS; this overwrites with real values.
        for name in agent_names:
            try:
                agent_data = strategy_chain_data.get(name)
                if not isinstance(agent_data, dict):
                    agent_data = {}

                # Extract web citations collected during tool execution
                raw_web = agent_data.pop("_web_citations", None)
                raw_web = raw_web if isinstance(raw_web, list) else []

                # Deduplicate by URL (same source may come from multiple tool calls)
                seen_urls: set = set()
                web: list = []
                for c in raw_web:
                    url = (c.get("url") or "") if isinstance(c, dict) else ""
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        web.append(c)

                kb = kb_citations_map.get(name, [])
                agent_data["citations"] = {"kb_sources": kb, "web_sources": web}
                strategy_chain_data[name] = agent_data
            except Exception as _ce:
                logger.warning(f"[Citations] failed to inject for {name}: {_ce}")

        for name in agent_names:
            AgentResponse.objects.filter(thread_id=thread_id, agent_name=name).update(
                status="complete",
                response=strategy_chain_data.get(name),
                runtime_ms=agents_runtime_ms,
            )
            emit(thread_id, f"agent_{name}", f"{name.replace('_', ' ').title()} — complete")

        # ── Aggregation ────────────────────────────────────────────
        emit(thread_id, "aggregating", "Aggregating final strategy brief...")
        agg = AggregatedStrategicAnalysis(objective=input_query)
        final = agg.make_aggregated_chains().invoke({
            "objective": input_query,
            **strategy_chain_data,
        })

        FinalStrategy.objects.update_or_create(
            thread_id=thread_id, defaults={"data": final}
        )

        total_ms = int((time.time() - start) * 1000)
        Thread.objects.filter(thread_id=thread_id).update(
            status='complete',
            runtime_ms=total_ms,
            completed_at=timezone.now(),
        )

        emit(thread_id, "done", "Strategy complete!", done=True, data={"thread_id": thread_id})

    except Exception as e:
        reason = str(e)
        logger.error(f"Pipeline failed for {thread_id}: {reason}", exc_info=True)
        Thread.objects.filter(thread_id=thread_id).update(status='failed', failed_reason=reason)
        emit(thread_id, "error", reason, done=True, error=True)
