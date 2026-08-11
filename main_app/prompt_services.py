from langchain_core.prompts import ChatPromptTemplate


class PromptService:
    def __init__(self):
        self.prompts = self.load_prompts()

    def load_prompts(self):
        # Prompts kept compact on purpose — free-tier Groq TPM is ~12k/min.
        EXECUTIVE_SUMMARY_PROMPT = """
You are a senior AI Strategy Consultant. Write a sharp Executive Summary from your knowledge.

Rules (keep answers short):
- problem_statement: 1-2 sentences
- strategic_opportunity: 1-2 sentences
- proposed_solution: 2 sentences max
- business_impact: 1-2 sentences with a number if possible
- key_market_opportunity: 1 line with size/growth signal
- time_to_market: 1 short phased timeline line
- key_highlights: exactly 3 short bullets
- confidence_score: integer 0-100

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON with keys:
{{"problem_statement":"...","strategic_opportunity":"...","proposed_solution":"...","business_impact":"...","key_market_opportunity":"...","time_to_market":"...","confidence_score":0,"key_highlights":["...","...","..."]}}
"""

        MARKET_ANALYSIS_PROMPT = """
You are a strategic market analyst. Estimate market intelligence from your knowledge. Every field required; keep each string concise (1-2 sentences).

Fields:
- market_size, market_growth_rate, market_opportunity, projected_size_5yr, market_overview
- key_trends, growth_drivers, key_challenges: exactly 3 short items each

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"market_size":"...","market_growth_rate":"...","market_opportunity":"...","projected_size_5yr":"...","market_overview":"...","key_trends":["...","...","..."],"growth_drivers":["...","...","..."],"key_challenges":["...","...","..."]}}
"""

        COMPETITIVE_LANDSCAPE_PROMPT = """
You are a competitive intelligence expert. Build a concise competitive analysis from your knowledge.

Rules:
- key_competitors: exactly 3 competitors; strengths/weaknesses as short strings
- porters_forces: score 1-5 each; overall_score average; industry_attractiveness High/Medium/Low; dominant_force name
- positioning_gaps: 2 short gaps
- our_advantages: 2 short advantages
- recommended_position: 1 sentence
- differentiation_strategy: 1 sentence

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"key_competitors":[{{"name":"...","strengths":["..."],"weaknesses":["..."],"market_position":"..."}}],"porters_forces":{{"supplier_power":1,"buyer_power":1,"competitive_rivalry":1,"threat_of_substitutes":1,"threat_of_new_entrants":1,"overall_score":1.0,"industry_attractiveness":"Medium","dominant_force":"..."}},"positioning_gaps":["...","..."],"our_advantages":["...","..."],"recommended_position":"...","differentiation_strategy":"..."}}
"""

        MONETIZATION_STRATEGY_PROMPT = """
You are a business model strategist. Design a concise monetization plan from industry benchmarks.

Include:
- recommended_pricing_model, pricing_model_score (0-100)
- revenue_streams: 2-3 short items
- unit_economics: arpu_usd, cac_usd, ltv_usd, ltv_cac_ratio, payback_months, health_grade (A-D)
- revenue_projection: year1_arr_usd, year2_arr_usd, year3_arr_usd
- customer_acquisition_strategy, scalability_notes: 1 sentence each

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"recommended_pricing_model":"...","pricing_model_score":0,"revenue_streams":["...","..."],"unit_economics":{{"arpu_usd":0,"cac_usd":0,"ltv_usd":0,"ltv_cac_ratio":0,"payback_months":0,"health_grade":"B"}},"revenue_projection":{{"year1_arr_usd":0,"year2_arr_usd":0,"year3_arr_usd":0}},"customer_acquisition_strategy":"...","scalability_notes":"..."}}
"""

        RISK_ASSESSMENT_PROMPT = """
You are a risk advisor. Identify key risks from your knowledge. Keep text short.

- risks: 3 items with category, description, probability 1-5, impact 1-5, risk_score, severity, mitigation, effort_weeks
- regulatory_risks: key_regulations (2-3), overall_risk_level, compliance_actions (2)
- top_risk: one line
- overall_risk_rating: Low|Medium|High|Critical

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"risks":[{{"category":"market","description":"...","probability":3,"impact":3,"risk_score":9,"severity":"Medium","mitigation":"...","effort_weeks":4}}],"regulatory_risks":{{"key_regulations":["...","..."],"overall_risk_level":"Medium","compliance_actions":["...","..."]}},"top_risk":"...","overall_risk_rating":"Medium"}}
"""

        ROADMAP_PROMPT = """
You are an execution strategist. Create a concise 30-60-90 day roadmap. Numbers must be plain integers (no arithmetic expressions).

Each phase: theme + 2 milestones with name, rice_score, priority_rank, weeks_estimate, cost_estimate_usd, risk_flag, success_metric.
Also: total_estimated_weeks, total_estimated_cost_usd, critical_path (1 line).

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"phases":[{{"phase":"30 days","theme":"...","milestones":[{{"name":"...","rice_score":100,"priority_rank":"Must Do","weeks_estimate":2,"cost_estimate_usd":7500,"risk_flag":"green","success_metric":"..."}}]}}],"total_estimated_weeks":12,"total_estimated_cost_usd":45000,"critical_path":"..."}}
"""

        WEAKNESS_REVIEW_PROMPT = """
You are a strategic auditor. List concise strategy weaknesses from your knowledge.

- weaknesses: 3 items with description, category, business_impact 1-5, fix_effort 1-5, quadrant, priority, recommendation
- dominant_pattern, root_cause (1 sentence), severity
- systemic_issues: 2 short items
- top_3_recommendations: 3 short items

Context:
{context}

Objective:
{objective}

Return ONLY valid JSON:
{{"weaknesses":[{{"description":"...","category":"execution","business_impact":3,"fix_effort":2,"quadrant":"Quick Win","priority":"High","recommendation":"..."}}],"dominant_pattern":"...","root_cause":"...","severity":"Medium","systemic_issues":["...","..."],"top_3_recommendations":["...","...","..."]}}
"""

        AGGREGATION_PROMPT = """
Synthesize one short strategy brief from these inputs.

Objective: {objective}
Inputs: {executive_summary} | {market_analysis} | {competitive_landscape} | {monetization_strategy} | {risk_assessment} | {roadmap} | {weakness_review}

Rules: executive_summary 2 sentences; key_insights 3 short bullets; critical_risks 2 with mitigation; recommendations 3; overall_feasibility High|Medium|Low; confidence_score 0-100.

Return ONLY valid JSON:
{{"executive_summary":"...","key_insights":["...","...","..."],"critical_risks":[{{"risk":"...","mitigation":"..."}},{{"risk":"...","mitigation":"..."}}],"recommendations":["...","...","..."],"overall_feasibility":"Medium","confidence_score":70}}
"""

        return {
            "EXECUTIVE_SUMMARY_PROMPT": EXECUTIVE_SUMMARY_PROMPT,
            "MARKET_ANALYSIS_PROMPT": MARKET_ANALYSIS_PROMPT,
            "COMPETITIVE_LANDSCAPE_PROMPT": COMPETITIVE_LANDSCAPE_PROMPT,
            "MONETIZATION_STRATEGY_PROMPT": MONETIZATION_STRATEGY_PROMPT,
            "RISK_ASSESSMENT_PROMPT": RISK_ASSESSMENT_PROMPT,
            "ROADMAP_PROMPT": ROADMAP_PROMPT,
            "WEAKNESS_REVIEW_PROMPT": WEAKNESS_REVIEW_PROMPT,
            "AGGREGATION_PROMPT": AGGREGATION_PROMPT,
        }

    def get_prompt(self, prompt_name):
        template = self.prompts.get(prompt_name)
        if not template:
            raise ValueError(f"No prompt template found for {prompt_name}")
        return ChatPromptTemplate.from_template(template)
