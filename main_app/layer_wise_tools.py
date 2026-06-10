from .tools import (
    MarketAnalysisTool,
    ExecutiveSummaryTools,
    CompetitiveLandscapeTools,
    RiskAssessmentTools,
)


class MarketLayerWiseTools:
    def __init__(self, thread_id=None):
        self.thread_id = thread_id
        self._market = MarketAnalysisTool()
        self._executive = ExecutiveSummaryTools(thread_id=thread_id)
        self._competitive = CompetitiveLandscapeTools()
        self._risk = RiskAssessmentTools()

    # ── Executive Summary ─────────────────────────────────────────────────────
    def get_executive_summary_tools(self):
        return [
            self._executive.executive_context_search(),
        ]

    # ── Market Analysis ───────────────────────────────────────────────────────
    def get_market_analysis_tools(self):
        return [
            self._market.MarketSearch(),
        ]

    # ── Competitive Landscape ─────────────────────────────────────────────────
    def get_competitive_landscape_tools(self):
        return [
            self._competitive.competitor_web_search(),
        ]

    # ── Monetization Strategy ─────────────────────────────────────────────────
    def get_monetization_tools(self):
        return []

    # ── Risk Assessment ───────────────────────────────────────────────────────
    def get_risk_assessment_tools(self):
        return [
            self._risk.regulatory_risk_checker(),
        ]

    # ── Product Roadmap ───────────────────────────────────────────────────────
    def get_roadmap_tools(self):
        return []

    # ── Weakness Review ───────────────────────────────────────────────────────
    def get_weakness_review_tools(self):
        return []
