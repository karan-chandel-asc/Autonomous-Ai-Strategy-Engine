import json
import re
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import ToolMessage, HumanMessage
from .prompt_services import PromptService
from .langchain_models import Lanchain_models
from .layer_wise_tools import MarketLayerWiseTools


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


_TOOL_RESULT_MAX_CHARS = 300


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
            response = bound_llm.invoke(base_messages)
        except Exception as exc:
            if "tool_use_failed" in str(exc) or "tool_use_failed" in repr(exc):
                response = base_llm.invoke(base_messages)
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
                            response = base_llm.invoke(base_messages + _synthetic_results)
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
            response = base_llm.invoke(fresh_messages)

        text = response.content if hasattr(response, "content") else str(response)
        text = _sanitize_json(text)

        stripped = text.strip()
        if stripped and not stripped.startswith("{") and not stripped.startswith("["):
            retry_messages = base_messages + [HumanMessage(content=_FORCE_JSON_MSG)]
            response = base_llm.invoke(retry_messages)
            text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

        # Parse JSON here so we can inject _web_citations before returning
        try:
            result_dict = json.loads(text)
            if not isinstance(result_dict, dict):
                result_dict = {}
        except (json.JSONDecodeError, ValueError):
            result_dict = {}

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
        self.tools = MarketLayerWiseTools(thread_id=thread_id)

    def _chain(self, prompt_key, tools_list, agent_name):
        context = self.contexts.get(agent_name, "")
        prompt = self.PromptService.get_prompt(prompt_key)
        chat_model = self.llm.get_chat_model()
        bound = chat_model.bind_tools(tools_list) if tools_list else chat_model
        by_name = {t.name: t for t in tools_list}
        inner = _build_agent_chain(prompt, bound, chat_model, by_name, self.parser)
        return RunnableLambda(lambda inp, ctx=context: inner.invoke({**inp, "context": ctx}))

    def make_parallel_chains(self):
        return RunnableParallel(
            executive_summary=self._chain(
                "EXECUTIVE_SUMMARY_PROMPT",
                self.tools.get_executive_summary_tools(),
                "executive_summary",
            ),
            market_analysis=self._chain(
                "MARKET_ANALYSIS_PROMPT",
                self.tools.get_market_analysis_tools(),
                "market_analysis",
            ),
            competitive_landscape=self._chain(
                "COMPETITIVE_LANDSCAPE_PROMPT",
                self.tools.get_competitive_landscape_tools(),
                "competitive_landscape",
            ),
            monetization_strategy=self._chain(
                "MONETIZATION_STRATEGY_PROMPT",
                self.tools.get_monetization_tools(),
                "monetization_strategy",
            ),
            risk_assessment=self._chain(
                "RISK_ASSESSMENT_PROMPT",
                self.tools.get_risk_assessment_tools(),
                "risk_assessment",
            ),
            roadmap=self._chain(
                "ROADMAP_PROMPT",
                self.tools.get_roadmap_tools(),
                "roadmap",
            ),
            weakness_review=self._chain(
                "WEAKNESS_REVIEW_PROMPT",
                self.tools.get_weakness_review_tools(),
                "weakness_review",
            ),
        )


class AggregatedStrategicAnalysis:
    def __init__(self, objective):
        self.objective = objective
        self.PromptService = PromptService()
        self.parser = JsonOutputParser()
        self.llm = Lanchain_models()

    def make_aggregated_chains(self):
        chat_model = self.llm.get_json_chat_model()
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
            response = chat_model.invoke(messages)
            text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

            stripped = text.strip()
            if stripped and not stripped.startswith("{") and not stripped.startswith("["):
                retry_messages = messages + [HumanMessage(content=_FORCE_JSON_MSG)]
                response = chat_model.invoke(retry_messages)
                text = _sanitize_json(response.content if hasattr(response, "content") else str(response))

            return parser.parse(text)

        return RunnableLambda(invoke)


# ── Schema defaults — every field the frontend reads must always be present ──

_AGENT_DEFAULTS = {
    "executive_summary": {
        "problem_statement":      "",
        "strategic_opportunity":  "",
        "proposed_solution":      "",
        "business_impact":        "",
        "key_market_opportunity": "",
        "time_to_market":         "",
        "confidence_score":       None,
        "key_highlights":         [],
        "citations":              {"kb_sources": [], "web_sources": []},
    },
    "market_analysis": {
        "market_size":        "",
        "market_growth_rate": "",
        "market_opportunity": "",
        "projected_size_5yr": "",
        "market_overview":    "",
        "key_trends":         [],
        "growth_drivers":     [],
        "key_challenges":     [],
        "citations":          {"kb_sources": [], "web_sources": []},
    },
    "competitive_landscape": {
        "key_competitors": [],
        "porters_forces": {
            "supplier_power":         None,
            "buyer_power":            None,
            "competitive_rivalry":    None,
            "threat_of_substitutes":  None,
            "threat_of_new_entrants": None,
            "overall_score":          None,
            "industry_attractiveness": None,
            "dominant_force":         None,
        },
        "positioning_gaps":         [],
        "our_advantages":           [],
        "recommended_position":     "",
        "differentiation_strategy": "",
        "citations":                {"kb_sources": [], "web_sources": []},
    },
    "monetization_strategy": {
        "recommended_pricing_model": "",
        "pricing_model_score":       None,
        "revenue_streams":           [],
        "unit_economics": {
            "arpu_usd":        None,
            "cac_usd":         None,
            "ltv_usd":         None,
            "ltv_cac_ratio":   None,
            "payback_months":  None,
            "health_grade":    None,
        },
        "revenue_projection": {
            "year1_arr_usd": None,
            "year2_arr_usd": None,
            "year3_arr_usd": None,
        },
        "customer_acquisition_strategy": "",
        "scalability_notes":             "",
        "citations":                     {"kb_sources": [], "web_sources": []},
    },
    "risk_assessment": {
        "risks": [],
        "regulatory_risks": {
            "key_regulations":    [],
            "overall_risk_level": None,
            "compliance_actions": [],
        },
        "top_risk":            "",
        "overall_risk_rating": None,
        "citations":           {"kb_sources": [], "web_sources": []},
    },
    "roadmap": {
        "phases":                   [],
        "total_estimated_weeks":    None,
        "total_estimated_cost_usd": None,
        "critical_path":            "",
        "citations":                {"kb_sources": [], "web_sources": []},
    },
    "weakness_review": {
        "weaknesses":            [],
        "dominant_pattern":      "",
        "root_cause":            "",
        "severity":              None,
        "systemic_issues":       [],
        "top_3_recommendations": [],
        "citations":             {"kb_sources": [], "web_sources": []},
    },
}


def _coerce_list(val):
    """Return val if it's a list, else wrap in list or return []."""
    if isinstance(val, list):
        return val
    if val is None:
        return []
    return [val] if isinstance(val, (str, dict)) else []


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
            # list field — coerce to list; keep LLM value if non-empty
            coerced = _coerce_list(av)
            out[key] = coerced if coerced else dv
        else:
            # scalar — use LLM value if present, otherwise default
            out[key] = av if av is not None else dv
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
        normalized[agent_name] = _merge(defaults, raw if isinstance(raw, dict) else {})
    return normalized
