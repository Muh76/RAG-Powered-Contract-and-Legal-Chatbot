# Legal Chatbot - Agentic RAG Service using LangChain
# Phase 3: Agentic RAG with tool calling and multi-step reasoning

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
import logging
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# LangChain 1.0+ imports
try:
    # Try LangChain 1.0+ (uses LangGraph)
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool
    LANGCHAIN_VERSION = "1.0+"
except ImportError:
    try:
        # Try LangChain 0.1.x style
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.tools import BaseTool
        LANGCHAIN_VERSION = "0.1.x"
    except ImportError:
        # Fallback to older imports
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import BaseTool
        LANGCHAIN_VERSION = "legacy"

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.tools.legal_search_tool import LegalSearchTool
from app.tools.statute_lookup_tool import StatuteLookupTool
from app.tools.document_analyzer_tool import DocumentAnalyzerTool
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Agentic RAG service with LangChain agents, tool calling, and multi-step reasoning.
    
    This service enables:
    - Autonomous tool selection based on query
    - Multi-step information gathering
    - Iterative refinement
    - Complex query handling that requires multiple searches
    """
    
    def __init__(self, mode: str = "solicitor"):
        """
        Initialize agentic RAG service.
        
        Args:
            mode: Response mode ("solicitor" or "public")
        """
        self.mode = mode
        self.rag_service = None
        self.llm_service = None
        self.agent_executor = None
        self.tools: List[BaseTool] = []
        
        # Agent configuration
        self.max_iterations = 5
        self.max_execution_time = 60  # seconds
        self.system_prompt = None  # Store system prompt for later use
        
        self._initialize()
    
    def _initialize(self):
        """Initialize agent components"""
        try:
            # Initialize RAG service with hybrid search
            logger.info("Initializing RAG service for agentic chatbot...")
            self.rag_service = RAGService(use_hybrid=True)
            logger.info("✅ RAG service initialized")
            
            # Initialize LLM service
            logger.info("Initializing LLM service...")
            self.llm_service = LLMService()
            logger.info("✅ LLM service initialized")
            
            # Initialize tools
            self._initialize_tools()
            
            # Initialize LangChain agent
            self._initialize_agent()
            
        except Exception as e:
            logger.error(f"Error initializing agentic RAG service: {e}", exc_info=True)
            raise
    
    def _initialize_tools(self):
        """Initialize LangChain tools"""
        try:
            # Legal search tool
            legal_search_tool = LegalSearchTool(rag_service=self.rag_service)
            self.tools.append(legal_search_tool)
            logger.info("✅ Legal search tool initialized")
            
            # Statute lookup tool
            statute_lookup_tool = StatuteLookupTool(rag_service=self.rag_service)
            self.tools.append(statute_lookup_tool)
            logger.info("✅ Statute lookup tool initialized")
            
            # Document analyzer tool (optional, requires LLM)
            try:
                document_analyzer_tool = DocumentAnalyzerTool(llm_service=self.llm_service)
                self.tools.append(document_analyzer_tool)
                logger.info("✅ Document analyzer tool initialized")
            except Exception as e:
                logger.warning(f"Document analyzer tool not available: {e}")
            
            logger.info(f"Initialized {len(self.tools)} tools for agent")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}", exc_info=True)
            raise
    
    def _initialize_agent(self):
        """Initialize LangChain agent with tools"""
        try:
            # Get OpenAI API key
            api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in settings or environment")
            
            # Initialize LLM with function calling support
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.1,  # Low temperature for consistent legal responses
                api_key=api_key
            )
            
            # Create system prompt based on mode
            if self.mode == "solicitor":
                self.system_prompt = """You are a legal assistant specializing in UK law with access to legal document search tools.

You must:
1. Use tools to search for relevant legal information when needed
2. Answer ONLY using information from the tools or your knowledge of UK law
3. Use precise legal terminology and cite specific sections
4. Include citations in format [1], [2], etc. for each claim from tool results
5. If tool results are insufficient, clearly state this
6. Maintain professional legal language

When you use tools:
- Use search_legal_documents for general legal queries
- Use get_specific_statute when the user mentions a specific Act or statute
- You can use multiple tools in sequence to gather comprehensive information
- Always cite sources from tool results

If the user's query is complex, break it down and use multiple tools to gather all necessary information."""
            else:  # public mode
                self.system_prompt = """You are a legal assistant helping the general public understand UK law with access to legal document search tools.

You must:
1. Use tools to search for relevant legal information when needed
2. Answer using information from the tools in plain language
3. Explain legal concepts clearly without jargon
4. Include citations in format [1], [2], etc. for each claim from tool results
5. If tool results are insufficient, clearly state this
6. Use accessible, everyday language

When you use tools:
- Use search_legal_documents for general legal queries
- Use get_specific_statute when the user mentions a specific Act or statute
- You can use multiple tools in sequence to gather comprehensive information
- Always cite sources from tool results

If the user's query is complex, break it down and use multiple tools to gather all necessary information."""

            # Initialize agent based on LangChain version
            if LANGCHAIN_VERSION == "1.0+":
                # Use LangGraph for LangChain 1.0+
                try:
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                    
                    # Create prompt with system message
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", self.system_prompt),
                        MessagesPlaceholder(variable_name="chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                    ])
                    
                    # Create agent using LangGraph with prompt
                    self.agent_executor = create_react_agent(
                        llm,
                        self.tools,
                        prompt=prompt
                    )
                except Exception as e:
                    logger.warning(f"Failed to create LangGraph agent with prompt: {e}, falling back to simple agent")
                    # Fallback to simple agent without custom prompt
                    self.agent_executor = create_react_agent(
                        llm,
                        self.tools
                    )
                
                logger.info("✅ LangGraph agent initialized (LangChain 1.0+)")
            else:
                # Use traditional LangChain agents (0.1.x or legacy)
                # Create prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
                
                # Create agent
                agent = create_openai_functions_agent(
                    llm=llm,
                    tools=self.tools,
                    prompt=prompt
                )
                
                # Create agent executor
                self.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    verbose=True,  # Enable verbose logging
                    max_iterations=self.max_iterations,
                    max_execution_time=self.max_execution_time,
                    return_intermediate_steps=True,  # Return tool call history
                    handle_parsing_errors=True  # Handle parsing errors gracefully
                )
                
                logger.info(f"✅ LangChain agent initialized ({LANGCHAIN_VERSION})")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain agent: {e}", exc_info=True)
            raise
    
    def chat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat query using agentic RAG with tool calling.
        
        Args:
            query: User query
            chat_history: Optional conversation history (list of {"role": "user/assistant", "content": "..."})
            
        Returns:
            Dictionary containing:
            - answer: Final answer text
            - tool_calls: List of tools used
            - iterations: Number of agent iterations
            - intermediate_steps: Agent execution steps
        """
        try:
            # Format chat history for LangChain
            formatted_history = []
            if chat_history:
                for msg in chat_history[-5:]:  # Keep last 5 messages for context
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted_history.append(("human", content))
                    elif role == "assistant":
                        formatted_history.append(("ai", content))
            
            # Invoke agent based on LangChain version
            if LANGCHAIN_VERSION == "1.0+":
                # LangGraph agent invocation (create_react_agent uses messages format)
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                
                # Build messages list
                messages = []
                # Add chat history if available
                if formatted_history:
                    for role, content in formatted_history:
                        if role == "human":
                            messages.append(HumanMessage(content=content))
                        elif role == "ai":
                            messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=query))
                
                config = {"recursion_limit": self.max_iterations}
                result = self.agent_executor.invoke({"messages": messages}, config=config)
                
                # Extract output from messages for LangGraph
                if isinstance(result, dict) and "messages" in result:
                    # Get the last message which should be the AI response
                    answer = result["messages"][-1].content if result["messages"] else "No response generated"
                else:
                    answer = str(result)
                intermediate_steps = []  # LangGraph handles this differently - would need stream events
            else:
                # Traditional agent invocation
                result = self.agent_executor.invoke({
                    "input": query,
                    "chat_history": formatted_history if formatted_history else []
                })
                answer = result.get("output", "I couldn't generate a response. Please try again.")
                intermediate_steps = result.get("intermediate_steps", [])
            
            # Track overall agent execution time before tool extraction
            import time
            agent_start_time = time.time()
            
            # Extract tool call information
            tool_calls = []
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_action = step[0]
                    tool_result = step[1]
                    
                    tool_name = tool_action.tool if hasattr(tool_action, "tool") else "unknown"
                    tool_input = tool_action.tool_input if hasattr(tool_action, "tool_input") else {}
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": str(tool_result)[:500]  # Limit result length
                    })
            
            # Track tool usage metrics
            if tool_calls:
                from app.core.metrics import metrics_collector
                agent_time_ms = (time.time() - agent_start_time) * 1000
                # Estimate time per tool (approximate - actual timing is tracked at tool level)
                estimated_time_per_tool = agent_time_ms / len(tool_calls) if len(tool_calls) > 0 else 0
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get("tool", "unknown")
                    # Track tool usage (success=True since we got results)
                    metrics_collector.record_tool_usage(
                        tool_name=tool_name,
                        execution_time_ms=estimated_time_per_tool,
                        success=True,
                    )
                
                logger.info(
                    f"Agent used {len(tool_calls)} tool(s)",
                    extra={
                        "tool_count": len(tool_calls),
                        "tool_names": [tc.get("tool") for tc in tool_calls],
                        "iterations": len(intermediate_steps),
                        "estimated_time_ms": round(estimated_time_per_tool, 2),
                        "type": "agent_tool_usage",
                    },
                )
            
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "iterations": len(intermediate_steps),
                "intermediate_steps_count": len(intermediate_steps),
                "mode": self.mode
            }
            
        except Exception as e:
            logger.error(f"Agent chat error: {e}", exc_info=True)
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "tool_calls": [],
                "iterations": 0,
                "intermediate_steps_count": 0,
                "mode": self.mode,
                "error": str(e)
            }
    
    async def achat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Async version of chat method"""
        # Run sync method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, query, chat_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agentic service"""
        return {
            "mode": self.mode,
            "tools_available": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "max_iterations": self.max_iterations,
            "max_execution_time": self.max_execution_time,
            "rag_service_ready": self.rag_service is not None,
            "llm_service_ready": self.llm_service is not None,
            "agent_ready": self.agent_executor is not None
        }

