import time
from langchain_core.tools import StructuredTool
from typing import Optional, List
from duckduckgo_search import DDGS
from .pydantic_schemas import *

_DDGS_TIMEOUT = 12    # seconds per DDGS request
_DDGS_RETRIES = 2     # attempts before giving up


def _ddgs_search(query: str, max_results: int) -> list:
    """DDGS wrapper with timeout + retry. Returns list of raw result dicts."""
    for attempt in range(_DDGS_RETRIES):
        try:
            with DDGS(timeout=_DDGS_TIMEOUT) as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception:
            if attempt < _DDGS_RETRIES - 1:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    return []


_REGULATORY_DB = {
    "fintech": [
        "PCI-DSS (global payment card security)",
        "PSD2 (EU open banking)",
        "FinCEN BSA / AML (US)",
        "FATF AML Guidelines (global)",
        "State Money Transmitter Licenses (US)",
        "FCA Authorization (UK)",
        "RBI Payment Aggregator Guidelines (India)",
        "MAS Payment Services Act (Singapore)",
    ],
    "healthtech": [
        "HIPAA (US health data privacy)",
        "GDPR – health data (EU)",
        "NHS DSP Toolkit (UK)",
        "FDA 21 CFR Part 11 (US electronic records)",
        "EU MDR (medical device regulation)",
        "ISO 13485 (medical devices QMS)",
        "IEC 62304 (medical device software lifecycle)",
    ],
    "saas": [
        "GDPR (EU data protection)",
        "CCPA / CPRA (California privacy)",
        "SOC 2 Type II (security audit standard)",
        "ISO 27001 (information security)",
        "CSA STAR (cloud security)",
    ],
    "marketplace": [
        "EU Digital Markets Act (DMA)",
        "EU Digital Services Act (DSA)",
        "Consumer Protection Act",
        "PCI-DSS (payment processing)",
        "GDPR (EU)",
    ],
    "edtech": [
        "FERPA (US student data privacy)",
        "COPPA (US children under 13)",
        "GDPR-K (EU children's data)",
        "WCAG 2.1 / ADA Section 508 (accessibility)",
    ],
    "ecommerce": [
        "GDPR cookie consent (EU)",
        "CCPA (California)",
        "PCI-DSS (payments)",
        "EU Consumer Rights Directive",
        "CAN-SPAM Act (US email marketing)",
    ],
}

_GEOGRAPHY_REGS = {
    "eu": ["EU AI Act 2024", "EU Data Act 2025", "GDPR"],
    "us": ["NIST Cybersecurity Framework", "FTC Act Section 5", "SOX (public cos)"],
    "uk": ["UK GDPR", "ICO Registration", "UK Cyber Essentials"],
    "india": ["DPDPA 2023", "IT Act 2000 (amended)", "MeitY guidelines"],
    "china": ["PIPL", "CSL (Cybersecurity Law)", "DSL (Data Security Law)", "MLPS 2.0"],
    "singapore": ["PDPA 2012 (amended 2020)", "MAS TRM Guidelines"],
    "australia": ["Privacy Act 1988", "Consumer Data Right (CDR)"],
    "canada": ["PIPEDA / Bill C-27 (CPPA)", "Quebec Law 25"],
}

_DATA_TYPE_REGS = {
    "pii": ["GDPR (EU)", "CCPA (US)", "DPDPA (India)"],
    "financial": ["PCI-DSS", "SOX", "PSD2 (EU)"],
    "health": ["HIPAA (US)", "GDPR – health (EU)", "ISO 27799"],
    "payment": ["PCI-DSS Level 1", "PSD2 SCA (EU)"],
    "biometric": ["BIPA (Illinois)", "GDPR – biometric (EU)"],
    "children": ["COPPA (US under-13)", "GDPR-K (EU)", "FERPA (educational)"],
}

_COMPLEXITY_BASE = {
    "low": (2, 0.5, 1.5),
    "medium": (6, 0.7, 1.4),
    "high": (12, 0.75, 1.5),
    "very_high": (24, 0.7, 1.6),
}

_WEAKNESS_PATTERNS = {
    "technical_debt": ["slow", "legacy", "outdated", "monolith", "scaling", "performance", "reliability", "tech debt"],
    "talent_gap": ["hiring", "skills", "expertise", "team", "leadership", "attrition", "burnout", "headcount"],
    "go_to_market": ["sales", "marketing", "acquisition", "brand", "distribution", "channel", "pipeline", "awareness"],
    "product_market_fit": ["retention", "churn", "engagement", "activation", "nps", "feedback", "repeat", "stickiness"],
    "operational": ["process", "efficiency", "cost", "overhead", "compliance", "quality", "support", "operations"],
    "financial": ["runway", "burn", "revenue", "cashflow", "profitability", "funding", "margin", "capital"],
    "strategic": ["vision", "roadmap", "focus", "direction", "prioritization", "competition", "differentiation", "clarity"],
}


class ToolClass:
    pass


class MarketAnalysisTool(ToolClass):
    def __init__(self):
        super().__init__()

    def fetch_market_data(self, industry: str, country: Optional[str] = None, focus: Optional[str] = None):
        parts = [industry]
        if country:
            parts.append(country)
        if focus:
            parts.append(focus)
        query = " ".join(parts)
        raw = _ddgs_search(query, max_results=2)
        results = [
            FetchMarketDataInput(title=r["title"], snippet=r["body"], link=r["href"])
            for r in raw
        ]
        return FetchMarketDataOutput(query=query, results=results)
        

    def MarketSearch(self):
        market_search_tool=StructuredTool.from_function(
            func=self.fetch_market_data,
            name="MarketSearch",
            description="Fetch market size, growth rate, forecast and main its govt policies.",
            args_schema=MarketDatalatestInputAnalysis
            )
        return market_search_tool
    



# ─── Executive Summary Tools ──────────────────────────────────────────────────
class ExecutiveSummaryTools(ToolClass):
    def __init__(self, thread_id=None):
        super().__init__()
        self.thread_id = thread_id

    # ── Executive context web search ─────────────────────────────────────────
    def _search_executive_context(self, company_or_industry: str, focus: Optional[str] = None) -> ExecutiveContextSearchOutput:
        query = f"{company_or_industry} {focus or 'strategy overview news funding'}"
        raw = _ddgs_search(query, max_results=2)
        results = [FetchMarketDataInput(title=r["title"], snippet=r["body"], link=r["href"]) for r in raw]
        return ExecutiveContextSearchOutput(query=query, results=results)

    def executive_context_search(self):
        return StructuredTool.from_function(
            func=self._search_executive_context,
            name="ExecutiveContextSearch",
            description="Search the web for the latest news, funding, leadership or strategic announcements for a company or industry.",
            args_schema=ExecutiveContextSearchInput,
        )


# ─── Competitive Landscape Tools ─────────────────────────────────────────────
class CompetitiveLandscapeTools(ToolClass):
    def __init__(self):
        super().__init__()

    # ── Competitor web search ────────────────────────────────────────────────
    def _search_competitors(self, industry: str, country: Optional[str] = None, focus: Optional[str] = None):
        parts = [industry, country or "", focus or "top competitors funding product comparison"]
        query = " ".join(p for p in parts if p)
        raw = _ddgs_search(query, max_results=3)
        results = [FetchMarketDataInput(title=r["title"], snippet=r["body"], link=r["href"]) for r in raw]
        return FetchMarketDataOutput(query=query, results=results)

    def competitor_web_search(self):
        return StructuredTool.from_function(
            func=self._search_competitors,
            name="CompetitorWebSearch",
            description="Search for top competitors, their funding, product features, and positioning in a given industry.",
            args_schema=CompetitorSearchInput,
        )



# ─── Risk Assessment Tools ────────────────────────────────────────────────────
class RiskAssessmentTools(ToolClass):
    def __init__(self):
        super().__init__()

    # ── Regulatory risk checker — real DB lookup ─────────────────────────────
    def _check_regulatory_risk(
        self,
        geographies: List[str],
        business_type: str,
        data_types_processed: List[str],
    ) -> RegulatoryRiskOutput:
        regs: set = set()

        biz = business_type.lower()
        for key, items in _REGULATORY_DB.items():
            if key in biz or biz in key:
                regs.update(items)

        for geo in geographies:
            g = geo.lower()
            for key, items in _GEOGRAPHY_REGS.items():
                if key in g or g in key:
                    regs.update(items)

        for dtype in data_types_processed:
            d = dtype.lower()
            for key, items in _DATA_TYPE_REGS.items():
                if key in d or d in key:
                    regs.update(items)

        count = len(regs)
        if count >= 8:
            risk_level = "Critical"
        elif count >= 5:
            risk_level = "High"
        elif count >= 3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        actions = [
            "Appoint a Data Protection Officer (DPO) if required",
            "Conduct a Data Protection Impact Assessment (DPIA)",
            "Implement a Privacy-by-Design framework",
            "Draft a compliant Privacy Policy and Terms of Service",
            "Set up a consent management platform (CMP)",
            "Establish incident response and breach notification procedures",
        ]

        return RegulatoryRiskOutput(
            key_regulations=sorted(regs)[:12],
            overall_risk_level=risk_level,
            compliance_actions=actions[:min(count + 2, 6)],
        )

    def regulatory_risk_checker(self):
        return StructuredTool.from_function(
            func=self._check_regulatory_risk,
            name="RegulatoryRiskChecker",
            description="Look up relevant regulations for given geographies, business type, and data types processed. Returns key regulations and compliance actions.",
            args_schema=RegulatoryRiskInput,
        )

