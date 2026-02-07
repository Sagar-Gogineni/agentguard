"""
AgentGuard + FastAPI Example

Deploy an EU AI Act compliant chatbot API in minutes.
All compliance (disclosure, audit, escalation) handled automatically.

Run:
    pip install fastapi uvicorn agentguard
    uvicorn fastapi_example:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agentguard import AgentGuard, EscalationTriggered

# ------------------------------------------------------------------ #
# Initialize AgentGuard
# ------------------------------------------------------------------ #

guard = AgentGuard(
    system_name="my-support-api",
    provider_name="my-provider",
    risk_level="limited",
    intended_purpose="Customer support chatbot API for product inquiries",
    disclosure_method="metadata",  # Return disclosure as JSON metadata
    audit_backend="sqlite",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "lawsuit", "medical", "refund dispute"],
    block_on_escalation=False,  # Set True to block responses that need human review
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    guard.close()


app = FastAPI(
    title="Support API (EU AI Act Compliant)",
    description="Powered by AgentGuard compliance middleware",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Your LLM function (replace with real implementation)
# ------------------------------------------------------------------ #


def call_llm(query: str) -> str:
    """Replace this with your actual LLM call."""
    return f"Here's information about: {query}"


# ------------------------------------------------------------------ #
# API Models
# ------------------------------------------------------------------ #


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    interaction_id: str
    ai_disclosure: dict
    content_label: dict | None = None
    escalated: bool = False


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with automatic EU AI Act compliance.

    Every response includes:
    - AI disclosure metadata (Article 50)
    - Content label (Article 50(2))
    - Audit trail entry (Article 12)
    - Human escalation check (Article 14)
    """
    try:
        result = guard.invoke(
            func=call_llm,
            input_text=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            model="gpt-4",
        )

        response = JSONResponse(
            content={
                "response": result["raw_response"],
                "interaction_id": result["interaction_id"],
                "ai_disclosure": result["disclosure"],
                "content_label": result["content_label"],
                "escalated": result["escalated"],
            },
            headers=result["disclosure"],  # Also set HTTP headers
        )
        return response

    except EscalationTriggered as e:
        raise HTTPException(
            status_code=202,
            detail={
                "message": "This query has been escalated for human review.",
                "interaction_id": e.interaction_id,
                "reason": e.reason,
            },
        )


@app.get("/compliance/report")
async def compliance_report():
    """Get the current compliance report."""
    return guard._reporter.generate_summary()


@app.get("/compliance/pending-reviews")
async def pending_reviews():
    """Get interactions pending human review (Article 14)."""
    return {"pending": guard.pending_reviews}


@app.post("/compliance/review/{interaction_id}/approve")
async def approve_review(interaction_id: str):
    """Approve a pending human review."""
    if guard.oversight.approve(interaction_id):
        return {"status": "approved"}
    raise HTTPException(404, "Interaction not found in review queue")


@app.post("/compliance/review/{interaction_id}/reject")
async def reject_review(interaction_id: str, reason: str = ""):
    """Reject a pending human review."""
    if guard.oversight.reject(interaction_id, reason):
        return {"status": "rejected", "reason": reason}
    raise HTTPException(404, "Interaction not found in review queue")
