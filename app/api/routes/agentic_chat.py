# Legal Chatbot - Agentic Chat Route
# Phase 3: Agentic RAG with LangChain

import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import (
    AgenticChatRequest, AgenticChatResponse, 
    ToolCall, SafetyReport, SafetyFlag, LatencyAndScores,
    Source
)
from app.services.agent_service import AgenticRAGService
from app.services.guardrails_service import GuardrailsService
from app.auth.dependencies import get_current_active_user, require_solicitor_or_admin
from app.auth.models import User, UserRole
from datetime import datetime
import time
import logging
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (singleton pattern)
agent_service_solicitor = None
agent_service_public = None
guardrails_service = None


def get_agent_service(mode: str = "public") -> AgenticRAGService:
    """Get or create agentic RAG service (singleton)"""
    global agent_service_solicitor, agent_service_public
    
    if mode == "solicitor":
        if agent_service_solicitor is None:
            logger.info("🔄 Initializing AgenticRAGService (solicitor mode)...")
            try:
                agent_service_solicitor = AgenticRAGService(mode="solicitor")
                logger.info("✅ AgenticRAGService (solicitor) initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize agentic service: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to initialize agentic service: {str(e)}")
        return agent_service_solicitor
    else:  # public mode
        if agent_service_public is None:
            logger.info("🔄 Initializing AgenticRAGService (public mode)...")
            try:
                agent_service_public = AgenticRAGService(mode="public")
                logger.info("✅ AgenticRAGService (public) initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize agentic service: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to initialize agentic service: {str(e)}")
        return agent_service_public


def get_guardrails_service():
    """Get or create guardrails service (singleton)"""
    global guardrails_service
    if guardrails_service is None:
        from app.services.guardrails_service import GuardrailsService
        guardrails_service = GuardrailsService()
        logger.info("✅ GuardrailsService initialized")
    return guardrails_service


def map_reason_to_safety_flag(reason: str) -> SafetyFlag:
    """Map guardrails reason to SafetyFlag enum"""
    reason_lower = reason.lower()
    if "harmful" in reason_lower or "injection" in reason_lower:
        return SafetyFlag.HARMFUL
    elif "domain" in reason_lower or "non_legal" in reason_lower or "insufficient_legal" in reason_lower:
        return SafetyFlag.NON_LEGAL
    elif "pii" in reason_lower or "personal" in reason_lower:
        return SafetyFlag.PII_DETECTED
    elif "injection" in reason_lower or "prompt" in reason_lower:
        return SafetyFlag.PROMPT_INJECTION
    else:
        return SafetyFlag.NON_LEGAL


@router.get("/agentic-chat")
async def agentic_chat_info():
    """Get agentic chat endpoint information"""
    return {
        "message": "Agentic chat endpoint - Use POST method",
        "method": "POST",
        "endpoint": "/api/v1/agentic-chat",
        "description": "Agentic chat with LangChain agents and tool calling",
        "requires_auth": True,
        "example_request": {
            "query": "What is the Sale of Goods Act 1979?",
            "mode": "public"
        },
        "docs": "/docs#/agentic-chat"
    }


@router.post("/agentic-chat", response_model=AgenticChatResponse)
async def agentic_chat(
    request: AgenticChatRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Agentic chat endpoint with LangChain agents and tool calling.
    Requires authentication.
    
    Role-based access:
    - Public users: Can only use "public" mode
    - Solicitor/Admin users: Can use both "public" and "solicitor" modes
    
    This endpoint enables:
    - Autonomous tool selection based on query
    - Multi-step information gathering
    - Iterative refinement
    - Complex query handling
    """
    start_time = time.time()
    
    # Role-based access control: Only Solicitor/Admin can use "solicitor" mode
    if request.mode.value == "solicitor" and current_user.role not in [UserRole.SOLICITOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Solicitor mode requires Solicitor or Admin role. Public users can only use 'public' mode."
        )
    
    try:
        # 1. Validate query with guardrails
        try:
            guardrails = get_guardrails_service()
        except Exception as e:
            logger.error(f"Guardrails service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Guardrails service error: {str(e)}")
        
        query_validation = guardrails.validate_query(request.query)
        
        if not query_validation["valid"]:
            safety_flag = map_reason_to_safety_flag(query_validation["reason"])
            
            return AgenticChatResponse(
                answer=query_validation["message"],
                tool_calls=[],
                iterations=0,
                intermediate_steps_count=0,
                mode=request.mode.value,
                safety=SafetyReport(
                    is_safe=False,
                    flags=[safety_flag],
                    confidence=0.9,
                    reasoning=query_validation.get("suggestion", "")
                ),
                metrics=LatencyAndScores(
                    retrieval_time_ms=0.0,
                    generation_time_ms=0.0,
                    total_time_ms=(time.time() - start_time) * 1000,
                    retrieval_score=0.0,
                    answer_relevance_score=0.0
                ),
                confidence_score=0.0,
                legal_jurisdiction="UK"
            )
        
        # 2. Get agent service
        try:
            agent_service = get_agent_service(mode=request.mode.value)
        except Exception as e:
            logger.error(f"Agent service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent service error: {str(e)}")
        
        # 3. Process query with agent (run in executor to avoid blocking)
        agent_start = time.time()
        try:
            result = await agent_service.achat(
                query=request.query,
                chat_history=request.chat_history
            )
        except Exception as agent_error:
            logger.error(f"Agent chat error: {agent_error}", exc_info=True)
            result = {
                "answer": "I encountered an error processing your query. Please try again.",
                "tool_calls": [],
                "iterations": 0,
                "intermediate_steps_count": 0,
                "mode": request.mode.value,
                "error": str(agent_error)
            }
        agent_time = (time.time() - agent_start) * 1000
        
        # 4. Format tool calls
        tool_calls = []
        for tool_call_dict in result.get("tool_calls", []):
            tool_calls.append(ToolCall(
                tool=tool_call_dict.get("tool", "unknown"),
                input=tool_call_dict.get("input", {}),
                result=tool_call_dict.get("result", "")
            ))
        
        # 5. Validate response
        response_validation = guardrails.validate_response({
            "answer": result.get("answer", ""),
            "tool_calls": tool_calls,
            "iterations": result.get("iterations", 0)
        })
        
        # 6. Calculate confidence score
        confidence_score = 0.8
        if result.get("iterations", 0) > 0:
            confidence_score = 0.85  # Agent used tools, likely more accurate
        if len(tool_calls) >= 2:
            confidence_score = min(confidence_score + 0.05, 0.95)  # Multiple tools used
        if not result.get("error"):
            confidence_score = min(confidence_score + 0.05, 0.95)
        
        # Map validation reason to SafetyFlag if needed
        safety_flags = []
        if not response_validation["valid"]:
            safety_flag = map_reason_to_safety_flag(response_validation["reason"])
            safety_flags = [safety_flag]
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        return AgenticChatResponse(
            answer=result.get("answer", "No response generated"),
            tool_calls=tool_calls,
            iterations=result.get("iterations", 0),
            intermediate_steps_count=result.get("intermediate_steps_count", 0),
            mode=result.get("mode", request.mode.value),
            safety=SafetyReport(
                is_safe=response_validation["valid"],
                flags=safety_flags,
                confidence=0.95 if response_validation["valid"] else 0.7,
                reasoning=response_validation["message"]
            ),
            metrics=LatencyAndScores(
                retrieval_time_ms=agent_time * 0.6,  # Estimate retrieval portion
                generation_time_ms=agent_time * 0.4,  # Estimate generation portion
                total_time_ms=total_time,
                retrieval_score=0.85 if tool_calls else 0.7,
                answer_relevance_score=confidence_score
            ),
            confidence_score=confidence_score,
            legal_jurisdiction="UK",
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agentic chat service error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agentic chat service error: {str(e)}")


@router.get("/agentic-chat/stats")
async def get_agent_stats():
    """Get statistics about the agentic service"""
    try:
        agent_service = get_agent_service(mode="public")
        stats = agent_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting agent stats: {str(e)}")

