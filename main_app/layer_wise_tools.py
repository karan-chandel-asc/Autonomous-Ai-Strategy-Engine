class MarketLayerWiseTools:
    """Tool bindings disabled to stay under Groq free-tier TPM (~12k)."""

    def __init__(self, thread_id=None):
        self.thread_id = thread_id

    def get_executive_summary_tools(self):
        return []

    def get_market_analysis_tools(self):
        return []

    def get_competitive_landscape_tools(self):
        return []

    def get_monetization_tools(self):
        return []

    def get_risk_assessment_tools(self):
        return []

    def get_roadmap_tools(self):
        return []

    def get_weakness_review_tools(self):
        return []
