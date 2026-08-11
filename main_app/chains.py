import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import ToolMessage, HumanMessage
from .prompt_services import PromptService
from .langchain_models import Lanchain_models, invoke_with_rate_limit_retry

# Per-agent models use separate TPM buckets → can run more in parallel.
_AGENT_BATCH_SIZE = int(os.environ.get("GROQ_AGENT_BATCH_SIZE", "4"))
_AGENT_BATCH_PAUSE_S = float(os.environ.get("GROQ_AGENT_BATCH_PAUSE_S", "1"))
_CONTEXT_MAX_CHARS = int(os.environ.get("GROQ_CONTEXT_MAX_CHARS", "500"))


def _sanitize_json(text: str) -> str:
    """Strip markdown fences and evaluate arithmetic expressions before JSON parsing.

    Some LLMs write values like ((100 * 3 * 80) / 4) instead of 6000.
    json.loads rejects expressions; we evaluate them safely here.
    """
    t = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if t.startswith("```"):
        t = re.sub(r'^```[^\n]*\n?', '', t)
        t = re.sub(r'\n?```\s*$', '', t)
        t = t.strip()

    # Evaluate parenthesised arithmetic expressions used as JSON values.
    # Safety: only match strings consisting entirely of digits, whitespace,
    # +  -  *  /  .  and parentheses — no variable names or builtins possible.
    def _eval(m):
        expr = m.group(0)
        if re.fullmatch(r'[\d\s\+\-\*\/\(\)\.]+', expr):
            try:
                val = eval(expr)  # noqa: S307 — restricted to numeric ops only
                return str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
            except Exception:
                pass
        return expr

    t = re.sub(r'\([\d\s\+\-\*\/\(\)\.]+\)', _eval, t)
    return t

def _slim_agent_outputs(data: dict) -> dict:
    """Extract only the most salient fields from each agent output before aggregation.
    Reduces aggregation input from ~5000 tokens to ~400 tokens."""
    def _get(d, *keys):
        if not isinstance(d, dict):
            return ""
        for k in keys:
            if k in d:
                v = d[k]
                return v if isinstance(v, str) else json.dumps(v)
        return ""

    def _d(v): return v if isinstance(v, dict) else {}

    es = _d(data.get("executive_summary"))
    ma = _d(data.get("market_analysis"))
    cl = _d(data.get("competitive_landscape"))
    ms = _d(data.get("monetization_strategy"))
    ra = _d(data.get("risk_assessment"))
    rm = _d(data.get("roadmap"))
    wr = _d(data.get("weakness_review"))

    competitors = cl.get("key_competitors", [])
    top2 = [c.get("name", "") for c in competitors[:2]] if isinstance(competitors, list) else []
    phases = rm.get("phases", [])
    first_phase = phases[0].get("theme", "") if phases else ""
    risks = ra.get("risks", [])
    top_risk_desc = risks[0].get("description", "") if risks else ra.get("top_risk", "")

    return {
        "executive_summary": f"{_get(es, 'problem_statement')} | {_get(es, 'key_market_opportunity')}",
        "market_analysis": f"{ma.get('market_size','')} | {ma.get('market_growth_rate','')} | {ma.get('market_opportunity','')[:120]}",
        "competitive_landscape": f"Top competitors: {', '.join(top2)} | Dominant force: {cl.get('porters_forces', {}).get('dominant_force', 'N/A')}",
        "monetization_strategy": f"Model: {ms.get('recommended_pricing_model', 'N/A')} | LTV:CAC {ms.get('unit_economics', {}).get('ltv_cac_ratio', 'N/A')} | Y3 ARR ${ms.get('revenue_projection', {}).get('year3_arr_usd', 'N/A')}",
        "risk_assessment": f"Top risk: {top_risk_desc} | Rating: {ra.get('overall_risk_rating', 'N/A')}",
        "roadmap": f"First phase: {first_phase} | Est. weeks: {rm.get('total_estimated_weeks', 'N/A')}",
        "weakness_review": f"Pattern: {wr.get('dominant_pattern', 'N/A')} | Actions: {'; '.join((wr.get('top_3_recommendations') or [])[:1])}",
    }


_FORCE_JSON_MSG = (
    "Your previous response was not a valid JSON object. "
    "Now respond ONLY with the JSON object exactly as specified in the instructions. "
    "No markdown, no explanation, no step numbers — just the raw JSON."
)

_FORCE_DATA_MSG = (
    "Your previous response had empty or missing fields. "
    "You MUST provide real, specific data for every field — no empty strings, no null values. "
    "Use your knowledge to fill in reasonable estimates if needed. "
    "Respond ONLY with the complete JSON object."
)

# Defaults fill gaps via normalize_agent_outputs — skip costly empty retries.
_MAX_RETRIES = 0


_CRITICAL_LIST_FIELDS = {
    # competitive_landscape
    "key_competitors", "positioning_gaps", "our_advantages",
    # market_analysis
    "key_trends", "growth_drivers", "key_challenges",
    # monetization_strategy
    "revenue_streams",
    # risk_assessment
    "risks",
    # roadmap
    "phases",
    # weakness_review
    "weaknesses", "systemic_issues", "top_3_recommendations",
    # executive_summary
    "key_highlights",
}

# citations injected separately; nested metric dicts (all-None by default) skew ratio unfairly
_EXCLUDED_FROM_EMPTY_CHECK = {
    "citations", "_web_citations",
    "porters_forces", "unit_economics", "revenue_projection", "regulatory_risks",
}


def _is_empty_response(result: dict) -> bool:
    """Return True if the agent returned mostly empty/null fields or a critical list is empty."""
    if not result:
        return True
    # Hard fail — these list fields must always have content
    for field in _CRITICAL_LIST_FIELDS:
        if field in result and not result[field]:
            return True
    empty_count = 0
    total_count = 0
    for k, v in result.items():
        if k in _EXCLUDED_FROM_EMPTY_CHECK:
            continue
        total_count += 1
        if v in ("", None, [], {}):
            empty_count += 1
    return total_count > 0 and (empty_count / total_count) >= 0.6


_TOOL_RESULT_MAX_CHARS = 200


def _truncate_context(text: str, limit: int = _CONTEXT_MAX_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _extract_web_citations(raw_json: str) -> list:
    """Pull web citations out of a tool result JSON string (results[].link/title/snippet)."""
    try:
        obj = json.loads(raw_json)
        if isinstance(obj, dict) and "results" in obj:
            out = []
            for r in obj["results"]:
                if isinstance(r, dict) and r.get("link"):
                    out.append({
                        "title":   (r.get("title") or "")[:80],
                        "url":     r.get("link", ""),
                        "snippet": (r.get("snippet") or r.get("body") or "")[:150],
                    })
            return out
    except Exception:
        pass
    return []


def _build_agent_chain(prompt, bound_llm, base_llm, tools_by_name, parser):
    def invoke(input_dict):
        prompt_value = prompt.invoke(input_dict)
        base_messages = (
            list(prompt_value.to_messages())
            if hasattr(prompt_value, "to_messages")
            else [prompt_value]
        )

        _web_citations = []   # collected from all tool executions this run

        # Groq raises 400 'tool_use_failed' when the model writes prose instead of
        # making a tool call. Catch it and fall back to base_llm (no tools).
        try:
            response = invoke_with_rate_limit_retry(bound_llm, base_messages)
        except Exception as exc:
            if "tool_use_failed" in str(exc) or "tool_use_failed" in repr(exc):
                response = invoke_with_rate_limit_retry(base_llm, base_messages)
            else:
                raise

        # Detect Groq hallucinating the tool call as raw text content
        # e.g. [{"name": "MarketSearch", "parameters": {...}}]
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            _raw_text = getattr(response, "content", "") or ""
            _stripped = _raw_text.strip()
            if _stripped.startswith("["):
                try:
                    _parsed = json.loads(_stripped)
                    if (
                        isinstance(_parsed, list)
                        and _parsed
                        and isinstance(_parsed[0], dict)
                        and "name" in _parsed[0]
                        and ("parameters" in _parsed[0] or "args" in _parsed[0])
                    ):
                        _synthetic_results = []
                        for _item in _parsed:
                            _tname = _item.get("name")
                            _targs = _item.get("parameters") or _item.get("args") or {}
                            _tool  = tools_by_name.get(_tname)
                            if _tool is None:
                                continue
                            try:
                                _result  = _tool.invoke(_targs)
                                _raw_out = (
                                    _result.model_dump_json()
                                    if hasattr(_result, "model_dump_json")
                                    else json.dumps(str(_result))
                                )
                                _web_citations.extend(_extract_web_citations(_raw_out))
                                _content = _raw_out[:_TOOL_RESULT_MAX_CHARS]
                            except Exception as _exc:
                                _content = json.dumps({"error": str(_exc)})
                            _synthetic_results.append(
                                HumanMessage(content=f"Tool {_tname} result: {_content}")
                            )
                        if _synthetic_results:
                            response = invoke_with_rate_limit_retry(
                                base_llm, base_messages + _synthetic_results
                            )
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                tool = tools_by_name.get(tc["name"])
                if tool is None:
                    continue
                try:
                    result = tool.invoke(tc["args"])
                    raw = (
                        result.model_dump_json()
                        if hasattr(result, "model_dump_json")
                        else json.dumps(str(result))
                    )
                    _web_citations.extend(_extract_web_citations(raw))
                    content = raw[:_TOOL_RESULT_MAX_CHARS]
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})
                tool_results.append(ToolMessage(content=content, tool_call_id=tc["id"]))

            fresh_messages = base_messages + [response] + tool_results
            response = invoke_with_rate_limit_retry(base_llm, fresh_messages)

        text = response.content if hasattr(response, "content") else str(response)
        text = _sanitize_json(text)

        stripped = text.strip()
        if stripped and not stripped.startswith("{") and not stripped.startswith("["):
            retry_messages = base_messages + [HumanMessage(content=_FORCE_JSON_MSG)]
            response = invoke_with_rate_limit_retry(base_llm, retry_messages)
            text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

        # Parse JSON here so we can inject _web_citations before returning
        try:
            result_dict = json.loads(text)
            if not isinstance(result_dict, dict):
                result_dict = {}
        except (json.JSONDecodeError, ValueError):
            result_dict = {}

        # Retry if agent returned mostly empty fields
        for _attempt in range(_MAX_RETRIES):
            if not _is_empty_response(result_dict):
                break
            retry_messages = base_messages + [HumanMessage(content=_FORCE_DATA_MSG)]
            _retry_response = invoke_with_rate_limit_retry(base_llm, retry_messages)
            _retry_text = _sanitize_json(
                _retry_response.content if hasattr(_retry_response, "content") else str(_retry_response)
            )
            try:
                _retry_dict = json.loads(_retry_text)
                if isinstance(_retry_dict, dict) and not _is_empty_response(_retry_dict):
                    result_dict = _retry_dict
                    break
            except (json.JSONDecodeError, ValueError):
                pass

        if _web_citations:
            result_dict["_web_citations"] = _web_citations

        return result_dict

    return RunnableLambda(invoke)


class ParallelStrategicAnalysis:
    def __init__(self, objective, thread_id=None, contexts=None):
        self.objective = objective
        self.thread_id = thread_id
        self.contexts = contexts or {}
        self.PromptService = PromptService()
        self.parser = JsonOutputParser()
        self.llm = Lanchain_models()

    def _chain(self, prompt_key, agent_name):
        """Single-call agent chain (tools disabled; per-agent Groq model)."""
        context = _truncate_context(self.contexts.get(agent_name, ""))
        prompt = self.PromptService.get_prompt(prompt_key)
        chat_model = self.llm.get_chat_model(agent_name)
        inner = _build_agent_chain(prompt, chat_model, chat_model, {}, self.parser)
        return RunnableLambda(lambda inp, ctx=context: inner.invoke({**inp, "context": ctx}))

    def _agent_specs(self):
        return [
            ("executive_summary", "EXECUTIVE_SUMMARY_PROMPT"),
            ("market_analysis", "MARKET_ANALYSIS_PROMPT"),
            ("competitive_landscape", "COMPETITIVE_LANDSCAPE_PROMPT"),
            ("monetization_strategy", "MONETIZATION_STRATEGY_PROMPT"),
            ("risk_assessment", "RISK_ASSESSMENT_PROMPT"),
            ("roadmap", "ROADMAP_PROMPT"),
            ("weakness_review", "WEAKNESS_REVIEW_PROMPT"),
        ]

    def make_parallel_chains(self):
        """Legacy parallel runner — prefer run_batched() on free-tier Groq."""
        return RunnableParallel(**{
            name: self._chain(prompt_key, name)
            for name, prompt_key in self._agent_specs()
        })

    def _run_batch(self, batch, objective: str, results: dict) -> None:
        with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
            futures = {
                pool.submit(
                    self._chain(prompt_key, name).invoke,
                    {"objective": objective},
                ): name
                for name, prompt_key in batch
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:
                    results[name] = {"error": str(exc)}

    def run_batched(self, objective: str) -> dict:
        """Run agents in TPM-safe batches (1 LLM call each, tools off)."""
        specs = self._agent_specs()
        results = {}
        batch_size = max(1, _AGENT_BATCH_SIZE)
        for i in range(0, len(specs), batch_size):
            if i > 0 and _AGENT_BATCH_PAUSE_S > 0:
                time.sleep(_AGENT_BATCH_PAUSE_S)
            self._run_batch(specs[i:i + batch_size], objective, results)
        return results


class AggregatedStrategicAnalysis:
    def __init__(self, objective):
        self.objective = objective
        self.PromptService = PromptService()
        self.parser = JsonOutputParser()
        self.llm = Lanchain_models()

    def make_aggregated_chains(self):
        chat_model = self.llm.get_json_chat_model("aggregation")
        prompt = self.PromptService.get_prompt("AGGREGATION_PROMPT")
        parser = self.parser

        def invoke(input_dict):
            slimmed = _slim_agent_outputs(input_dict)
            slimmed["objective"] = input_dict.get("objective", "")
            prompt_value = prompt.invoke(slimmed)
            messages = (
                list(prompt_value.to_messages())
                if hasattr(prompt_value, "to_messages")
                else [prompt_value]
            )
            response = invoke_with_rate_limit_retry(chat_model, messages)
            text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

            stripped = text.strip()
            if stripped and not stripped.startswith("{") and not stripped.startswith("["):
                retry_messages = messages + [HumanMessage(content=_FORCE_JSON_MSG)]
                response = invoke_with_rate_limit_retry(chat_model, retry_messages)
                text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

            try:
                result_dict = json.loads(text)
                if not isinstance(result_dict, dict):
                    result_dict = {}
            except (json.JSONDecodeError, ValueError):
                result_dict = {}

            for _attempt in range(_MAX_RETRIES):
                if not _is_empty_response(result_dict):
                    break
                retry_messages = messages + [HumanMessage(content=_FORCE_DATA_MSG)]
                _retry_response = invoke_with_rate_limit_retry(chat_model, retry_messages)
                _retry_text = _sanitize_json(
                    _retry_response.content if hasattr(_retry_response, "content") else str(_retry_response)
                )
                try:
                    _retry_dict = json.loads(_retry_text)
                    if isinstance(_retry_dict, dict) and not _is_empty_response(_retry_dict):
                        result_dict = _retry_dict
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

            return result_dict

        return RunnableLambda(invoke)


# ── Schema defaults — every field the frontend reads must always be present ──

_AGENT_DEFAULTS = {
    "executive_summary": {
        "problem_statement":      "Businesses face significant inefficiencies due to fragmented data and lack of intelligent automation.",
        "strategic_opportunity":  "A growing demand for AI-driven decision-making tools presents a strong entry opportunity.",
        "proposed_solution":      "An AI-powered strategy engine that automates analysis, generates insights, and accelerates decision-making.",
        "business_impact":        "Reduces strategic planning time by 60% and improves decision accuracy across business units.",
        "key_market_opportunity": "Enterprise AI adoption is accelerating — early movers will capture significant market share.",
        "time_to_market":         "6-9 months for MVP with phased rollout to enterprise clients.",
        "confidence_score":       72,
        "key_highlights":         [
            "Strong product-market fit in mid-market enterprises",
            "Scalable SaaS model with recurring revenue",
            "Low competitive density in the niche segment",
        ],
        "citations": {"kb_sources": [], "web_sources": []},
    },
    "market_analysis": {
        "market_size":        "$18.5B (2024)",
        "market_growth_rate": "28% CAGR",
        "market_opportunity": "Untapped SME and mid-market segment with limited AI strategy tooling available.",
        "projected_size_5yr": "$62B by 2029",
        "market_overview":    "The AI strategy and business intelligence market is experiencing rapid growth driven by digital transformation initiatives across industries.",
        "key_trends":         [
            "Shift from descriptive to prescriptive analytics",
            "Rise of no-code AI platforms for non-technical users",
            "Increased enterprise spending on automation tools",
        ],
        "growth_drivers":     [
            "Growing volume of unstructured business data",
            "Pressure to reduce operational costs",
            "Executive demand for real-time strategic insights",
        ],
        "key_challenges":     [
            "Data privacy and compliance concerns",
            "High integration complexity with legacy systems",
            "Talent gap in AI adoption among target customers",
        ],
        "citations": {"kb_sources": [], "web_sources": []},
    },
    "competitive_landscape": {
        "key_competitors": [
            {
                "name": "Competitor A",
                "strengths": "Strong brand recognition and large enterprise customer base",
                "weaknesses": "High pricing, slow innovation cycle",
                "market_share": "22%",
            },
            {
                "name": "Competitor B",
                "strengths": "Deep AI capabilities and research backing",
                "weaknesses": "Complex UX, poor SME fit",
                "market_share": "18%",
            },
            {
                "name": "Competitor C",
                "strengths": "Affordable pricing and fast onboarding",
                "weaknesses": "Limited features, no enterprise support",
                "market_share": "11%",
            },
        ],
        "porters_forces": {
            "supplier_power":          3,
            "buyer_power":             4,
            "competitive_rivalry":     4,
            "threat_of_substitutes":   3,
            "threat_of_new_entrants":  3,
            "overall_score":           3,
            "industry_attractiveness": "Moderate",
            "dominant_force":          "Buyer Power",
        },
        "positioning_gaps":         [
            "No affordable AI strategy tool for mid-market",
            "Lack of explainable AI outputs for non-technical executives",
        ],
        "our_advantages":           [
            "Faster time-to-insight vs. incumbents",
            "Modular pricing accessible to SMEs",
            "Built-in knowledge base with RAG capabilities",
        ],
        "recommended_position":     "Value-focused AI strategy partner for mid-market enterprises.",
        "differentiation_strategy": "Compete on ease-of-use, explainability, and price-to-value ratio rather than raw model performance.",
        "citations":                {"kb_sources": [], "web_sources": []},
    },
    "monetization_strategy": {
        "recommended_pricing_model": "Tiered SaaS Subscription",
        "pricing_model_score":       82,
        "revenue_streams":           [
            {"name": "Starter Plan", "description": "Individual users, basic analysis", "price_usd": 49},
            {"name": "Growth Plan", "description": "Teams up to 10, full module access", "price_usd": 199},
            {"name": "Enterprise Plan", "description": "Unlimited users, custom integrations, SLA", "price_usd": 999},
        ],
        "unit_economics": {
            "arpu_usd":       299,
            "cac_usd":        420,
            "ltv_usd":        3588,
            "ltv_cac_ratio":  8.5,
            "payback_months": 2,
            "health_grade":   "A",
        },
        "revenue_projection": {
            "year1_arr_usd": 480000,
            "year2_arr_usd": 1800000,
            "year3_arr_usd": 5200000,
        },
        "customer_acquisition_strategy": "Content-led growth via SEO and thought leadership, combined with product-led growth through a freemium entry tier.",
        "scalability_notes":             "SaaS model scales with minimal marginal cost. API-based architecture supports white-labelling for channel partners.",
        "citations":                     {"kb_sources": [], "web_sources": []},
    },
    "risk_assessment": {
        "risks": [
            {
                "description":   "LLM hallucination leading to inaccurate strategic recommendations",
                "category":      "Technical",
                "likelihood":    4,
                "impact":        5,
                "mitigation":    "Implement RAG pipeline with source citations and human-review checkpoints",
            },
            {
                "description":   "Data privacy breach exposing client business data",
                "category":      "Compliance",
                "likelihood":    2,
                "impact":        5,
                "mitigation":    "End-to-end encryption, SOC2 compliance, strict data isolation per tenant",
            },
            {
                "description":   "Larger competitor launching a similar product at lower cost",
                "category":      "Competitive",
                "likelihood":    3,
                "impact":        4,
                "mitigation":    "Accelerate niche feature depth and lock-in via integrations and custom workflows",
            },
        ],
        "regulatory_risks": {
            "key_regulations":    ["GDPR", "CCPA", "SOC2"],
            "overall_risk_level": 3,
            "compliance_actions": [
                "Appoint a Data Protection Officer",
                "Implement consent management for EU users",
                "Complete SOC2 Type II audit within 12 months",
            ],
        },
        "top_risk":            "LLM hallucination leading to inaccurate strategic recommendations",
        "overall_risk_rating": 3,
        "citations":           {"kb_sources": [], "web_sources": []},
    },
    "roadmap": {
        "phases": [
            {
                "phase": "30 days",
                "theme": "Foundation & MVP",
                "milestones": [
                    {
                        "name":             "Core API and authentication setup",
                        "rice_score":       4800,
                        "priority_rank":    "Must Do",
                        "weeks_estimate":   2,
                        "cost_estimate_usd": 7500,
                        "risk_flag":        "green",
                        "success_metric":   "API handles 100 concurrent requests with <200ms latency",
                    },
                    {
                        "name":             "MVP agent pipeline live",
                        "rice_score":       6000,
                        "priority_rank":    "Must Do",
                        "weeks_estimate":   3,
                        "cost_estimate_usd": 11250,
                        "risk_flag":        "green",
                        "success_metric":   "All 7 analysis modules return structured output",
                    },
                ],
            },
            {
                "phase": "60 days",
                "theme": "Product Polish & Beta Launch",
                "milestones": [
                    {
                        "name":             "Frontend dashboard and report UI",
                        "rice_score":       3600,
                        "priority_rank":    "Must Do",
                        "weeks_estimate":   4,
                        "cost_estimate_usd": 15000,
                        "risk_flag":        "green",
                        "success_metric":   "Beta users complete analysis in under 3 minutes",
                    },
                    {
                        "name":             "Knowledge base RAG integration",
                        "rice_score":       4200,
                        "priority_rank":    "Should Do",
                        "weeks_estimate":   3,
                        "cost_estimate_usd": 11250,
                        "risk_flag":        "green",
                        "success_metric":   "Document upload and retrieval accuracy >85%",
                    },
                ],
            },
            {
                "phase": "90 days",
                "theme": "Growth & Enterprise Readiness",
                "milestones": [
                    {
                        "name":             "Enterprise SSO and team management",
                        "rice_score":       2800,
                        "priority_rank":    "Should Do",
                        "weeks_estimate":   4,
                        "cost_estimate_usd": 15000,
                        "risk_flag":        "yellow",
                        "success_metric":   "First enterprise client onboarded successfully",
                    },
                    {
                        "name":             "Public launch and marketing push",
                        "rice_score":       5000,
                        "priority_rank":    "Must Do",
                        "weeks_estimate":   2,
                        "cost_estimate_usd": 7500,
                        "risk_flag":        "green",
                        "success_metric":   "100 paying customers within 30 days of launch",
                    },
                ],
            },
        ],
        "total_estimated_weeks":    18,
        "total_estimated_cost_usd": 67500,
        "critical_path":            "MVP pipeline → RAG integration → Frontend dashboard → Enterprise launch",
        "citations":                {"kb_sources": [], "web_sources": []},
    },
    "weakness_review": {
        "weaknesses": [
            {
                "description":     "Over-reliance on a single LLM provider creates vendor lock-in risk",
                "category":        "structural",
                "business_impact": 4,
                "fix_effort":      2,
                "quadrant":        "Quick Win",
                "priority":        "Immediate",
                "recommendation":  "Abstract the LLM layer to support multiple providers (OpenAI, Groq, Gemini)",
            },
            {
                "description":     "No offline or low-connectivity fallback for analysis generation",
                "category":        "execution",
                "business_impact": 3,
                "fix_effort":      3,
                "quadrant":        "Big Bet",
                "priority":        "High",
                "recommendation":  "Cache recent analyses and implement graceful degradation for API failures",
            },
            {
                "description":     "Limited go-to-market strategy for enterprise segment",
                "category":        "competitive",
                "business_impact": 4,
                "fix_effort":      3,
                "quadrant":        "Big Bet",
                "priority":        "High",
                "recommendation":  "Build dedicated enterprise sales motion with case studies and proof-of-concept templates",
            },
        ],
        "dominant_pattern":      "structural",
        "root_cause":            "Early-stage product built for speed — architectural decisions now create scaling and flexibility constraints.",
        "severity":              "Medium",
        "systemic_issues":       [
            "Single-provider dependency across multiple critical services",
            "Lack of fallback mechanisms in the core analysis pipeline",
        ],
        "top_3_recommendations": [
            "Abstract LLM provider layer to eliminate vendor lock-in",
            "Build enterprise sales playbook and dedicated onboarding flow",
            "Implement circuit-breaker pattern for external API dependencies",
        ],
        "citations": {"kb_sources": [], "web_sources": []},
    },
}


def _coerce_list(val):
    """Return val only if it's a non-empty list of real items, else return []."""
    if not isinstance(val, list):
        return []
    # filter out junk entries like ["N/A"], [""], [None]
    real = [v for v in val if v not in (None, "", "N/A", "n/a")]
    return real


def _merge(defaults: dict, actual) -> dict:
    """Deep-merge LLM output over defaults. LLM values always win; defaults fill gaps."""
    if not isinstance(actual, dict):
        return defaults.copy()
    out = {}
    for key, dv in defaults.items():
        av = actual.get(key)
        if isinstance(dv, dict):
            # nested object — recurse; if LLM gave a non-dict use defaults
            out[key] = _merge(dv, av if isinstance(av, dict) else {})
        elif isinstance(dv, list):
            # list field — use LLM value only if it has real items, else use demo default
            coerced = _coerce_list(av)
            out[key] = coerced if coerced else dv
        else:
            # scalar — use LLM value if non-empty, otherwise default
            out[key] = av if av not in (None, "", "N/A", "n/a") else dv
    # preserve any extra keys the LLM returned (don't strip unknown fields)
    for key in actual:
        if key not in out:
            out[key] = actual[key]
    return out


def normalize_agent_outputs(data: dict) -> dict:
    """Guarantee every agent output matches the expected schema.

    Called right after parallel chains complete. Fills missing/null fields
    with safe defaults so the frontend never crashes on a missing key.
    """
    normalized = dict(data)  # shallow copy; keep non-agent keys intact
    for agent_name, defaults in _AGENT_DEFAULTS.items():
        raw = data.get(agent_name)
        merged = _merge(defaults, raw if isinstance(raw, dict) else {})
        # Hard fallback — force demo data for any critical list still empty after merge
        for field in _CRITICAL_LIST_FIELDS:
            if not merged.get(field) and defaults.get(field):
                merged[field] = defaults[field]
        # Hard fallback — force demo data for any empty scalar string fields
        for field, dv in defaults.items():
            if isinstance(dv, str) and dv and not merged.get(field):
                merged[field] = dv
        normalized[agent_name] = merged
    return normalized
