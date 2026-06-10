"""The user is asking about what things to keep in mind when creating agents like the ones in this project. This is a conceptual/educational question, not a code change request. They want to understand the key considerations for building parallel AI agents with LangChain/Groq.

Let me think about what's relevant given their specific setup:

Rate limits (TPM/RPM)
Token budgeting
Tool design
Prompt design for JSON output
Error handling
Parallelism considerations
Model selection
Output validation
Retry logic and backoff strategies
I should give a concise, practical answer focused on their specific context with Groq and LangChain's RunnableParallel.

Given exactly what you built here, these are the things that will bite you if you ignore them:

1. Rate Limits — plan before you parallelize

Know your TPM (tokens/min) and RPM (requests/min) limits upfront
RunnableParallel fires everything at once — 7 agents × 2 calls each = 14 requests in 1 second
Rule of thumb: (num_agents × avg_tokens_per_agent) < 70% of TPM limit
2. Token Budget per Agent

Input tokens = prompt + tool schemas + any context passed in
Tool schemas can add 400–800 tokens per request — invisible but expensive
Set max_tokens per agent only as high as the output actually needs — not all agents need 1500
3. Tools — only give what's needed

Every tool you bind adds its full JSON schema to every request
If an agent doesn't need web search, don't give it web search
Fewer tools = smaller input = faster + cheaper + fewer tool_use_failed errors
4. Always force JSON output explicitly

In the prompt: "Return ONLY a valid JSON object, no markdown"
At the API level: response_format: {"type": "json_object"} (Groq JSON mode)
Have a retry fallback for when the model still returns prose
Without all three layers, you'll get markdown 20–30% of the time
5. Error handling per chain, not just globally

RunnableParallel will crash the entire run if one chain throws
Wrap each chain's invoke in try/except so one bad agent doesn't kill all 7
Groq-specific: catch tool_use_failed (model wrote prose instead of calling the tool) and fall back to direct generation
6. Prompt must match output schema exactly

Your parser (JsonOutputParser) expects specific keys
If the prompt says "write an analysis" but the parser expects {"market_size": ...}, it will always fail
Define the exact JSON schema in the prompt, key by key
7. Model limits vs task complexity

Smaller/faster models (like llama-4-scout) struggle with tool calling + complex JSON simultaneously
The more you ask in one prompt, the more likely it hallucinates structure
Split complex tasks: one call to gather data (tools), separate call to format output (JSON)
8. Aggregation is the most token-heavy call

It receives all 7 agents' outputs as input — easily 5,000–8,000 input tokens alone
This is the one most likely to hit rate limits or truncate
Keep individual agent outputs concise so aggregation input stays manageable
The mental model:


Per agent cost = prompt tokens + tool schema tokens + output tokens + (retry tokens if any)
Total run cost = (per agent cost × num agents) + aggregation input + aggregation output
Must be < TPM limit in one burst"""