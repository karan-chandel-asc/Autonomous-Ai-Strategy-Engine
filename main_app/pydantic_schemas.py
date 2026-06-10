from pydantic import BaseModel, Field, field_validator, BeforeValidator, validator
from typing import Optional, List, Any, Union, Annotated

def _to_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return v

def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v

# Groq validates tool schemas strictly — these types accept both string and number
CoercibleInt   = Annotated[Union[int,   str], BeforeValidator(_to_int)]
CoercibleFloat = Annotated[Union[float, str], BeforeValidator(_to_float)]
# step2 market
class MarketDatalatestInputAnalysis(BaseModel):
    industry: str = Field(..., description="Target industry name")
    country: Optional[str] = Field(None, description="Target country")
    focus: Optional[str] = Field(
        None,
        description="Specific focus like market size, growth rate, policy, forecast"
    )

class MarketDatalatestOutputAnalysis(BaseModel):
    market_size: Optional[str]
    growth_rate: Optional[str]
    forecast: Optional[str]
    key_policies: Optional[List[str]]
    source: Optional[str]
    year: Optional[int]

class FetchMarketDataInput(BaseModel):
    title: str = Field(description="Title in max 30 words")
    snippet: str = Field(description="Summary in max 300 words")
    link: str


class FetchMarketDataOutput(BaseModel):
    query: str
    results: List[FetchMarketDataInput]



# ─── Executive Summary ────────────────────────────────────────────────────────
class ExecutiveContextSearchInput(BaseModel):
    company_or_industry: str = Field(..., description="Company name or industry to search for")
    focus: Optional[str] = Field(None, description="Specific angle: strategy, funding, leadership, news")

class ExecutiveContextSearchOutput(BaseModel):
    query: str
    results: List["FetchMarketDataInput"]


# ─── Competitive Landscape ────────────────────────────────────────────────────
class CompetitorSearchInput(BaseModel):
    industry: str = Field(..., description="Industry or niche to search competitor data for")
    country: Optional[str] = Field(None)
    focus: Optional[str] = Field(None, description="e.g. pricing, features, funding, market share")



# ─── Risk Assessment ──────────────────────────────────────────────────────────
class RegulatoryRiskInput(BaseModel):
    geographies: List[str] = Field(..., description="Target countries or regions, e.g. US, EU, India")
    business_type: str = Field(..., description="e.g. fintech, healthtech, saas, marketplace, edtech")
    data_types_processed: List[str] = Field(..., description="e.g. PII, financial, health, payment, biometric")

class RegulatoryRiskOutput(BaseModel):
    key_regulations: List[str]
    overall_risk_level: str
    compliance_actions: List[str]


class InputValidation(BaseModel):
    thread_id: Optional[str] = None
    documents: Optional[List[Any]] = None
    input_query: str

    @validator("input_query")
    def validate_input_query(cls, value):
        if not value.strip():
            raise ValueError("input_query cannot be empty")
        return value

    @validator("thread_id")
    def validate_thread_id(cls, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("thread_id must be a string")
        return value

    @validator("documents")
    def validate_documents(cls, documents):
        if not documents:
            return documents

        if len(documents) > 10:
            raise ValueError(
                "A maximum of 10 documents can be uploaded at a time."
            )

        max_size = 50 * 1024 * 1024  # 50 MB

        for document in documents:
            if document.size > max_size:
                raise ValueError(
                    f"'{document.name}' exceeds the maximum file size limit of 50 MB."
                )

        return documents
    