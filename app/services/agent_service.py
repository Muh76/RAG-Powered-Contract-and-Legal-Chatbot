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

CRITICAL: When tool results contain "COMPLETE SECTION TEXT FROM KNOWLEDGE BASE" or "ACTUAL TEXT from the statutes", 
you MUST use that content directly in your answer. Do NOT say "the text does not elaborate" or "details were not provided" 
if the tool result contains actual legal text - use it verbatim or paraphrase it accurately.

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

CRITICAL: When tool results contain "COMPLETE SECTION TEXT FROM KNOWLEDGE BASE" or "ACTUAL TEXT from the statutes", 
you MUST use that content directly in your answer. Do NOT say "the text does not elaborate" or "details were not provided" 
if the tool result contains actual legal text - use it verbatim or paraphrase it accurately in plain language.

When you use tools:
- Use search_legal_documents for general legal queries
- Use get_specific_statute when the user mentions a specific Act or statute
- You can use multiple tools in sequence to gather comprehensive information
- Always cite sources from tool results

If the user's query is complex, break it down and use multiple tools to gather all necessary information."""
            
            # Initialize LLM with function calling support
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.1,  # Low temperature for consistent legal responses
                api_key=api_key
            )

            # Initialize agent based on LangChain version
            if LANGCHAIN_VERSION == "1.0+":
                # Use LangGraph for LangChain 1.0+
                # Try newer create_agent first, fallback to create_react_agent
                try:
                    # CRITICAL FIX: Bind system message to LLM instead of adding as SystemMessage
                    # create_react_agent filters SystemMessage before sending to OpenAI, causing empty array error
                    try:
                        llm_with_system = llm.bind(system=self.system_prompt)
                        logger.info("✅ System message bound to LLM")
                        self._use_bound_system = True
                    except Exception as bind_error:
                        logger.warning(f"Could not bind system message to LLM: {bind_error}")
                        logger.info("Will add SystemMessage to messages array instead")
                        llm_with_system = llm
                        self._use_bound_system = False
                    
                    # Try create_react_agent first (more reliable)
                    try:
                        from langgraph.prebuilt import create_react_agent
                        logger.info("Using create_react_agent")
                        self.agent_executor = create_react_agent(llm_with_system, self.tools)
                        self._agent_type = "create_react_agent"
                    except (ImportError, AttributeError):
                        # Fallback to create_agent from langchain.agents
                        try:
                            from langchain.agents import create_agent
                            logger.info("Falling back to create_agent from langchain.agents")
                            self.agent_executor = create_agent(llm_with_system, self.tools)
                            self._agent_type = "create_agent"
                        except (ImportError, AttributeError):
                            raise ImportError("Neither create_react_agent nor create_agent available")
                    
                    # Store system prompt for fallback (if binding failed)
                    self._system_prompt = self.system_prompt
                    logger.info(f"✅ LangGraph agent initialized (type: {self._agent_type}, system_bound={self._use_bound_system})")
                except Exception as e:
                    logger.error(f"Failed to create LangGraph agent: {e}", exc_info=True)
                    raise
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
                # LangGraph agent invocation with comprehensive logging
                config = {"recursion_limit": self.max_iterations}
                
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                
                # Build messages list with detailed logging
                messages = []
                logger.info(f"🔍 Building messages for query: '{query[:50]}...'")
                
                # Add system message only if NOT bound to LLM
                # If system is bound to LLM, don't add SystemMessage (it causes empty array error)
                if not getattr(self, '_use_bound_system', False):
                    if hasattr(self, '_system_prompt') and self._system_prompt:
                        system_msg = SystemMessage(content=self._system_prompt)
                        messages.append(system_msg)
                        logger.debug(f"  ✅ Added system message ({len(self._system_prompt)} chars)")
                else:
                    logger.debug(f"  ✅ System message already bound to LLM (not adding SystemMessage)")
                
                # Add chat history if available
                if formatted_history:
                    logger.debug(f"  📜 Adding {len(formatted_history)} history messages")
                    for role, content in formatted_history:
                        if role == "human":
                            messages.append(HumanMessage(content=content))
                        elif role == "ai":
                            messages.append(AIMessage(content=content))
                
                # Always add the query as a human message (must be last)
                if query and query.strip():
                    query_msg = HumanMessage(content=query)
                    messages.append(query_msg)
                    logger.debug(f"  ✅ Added query message ({len(query)} chars)")
                else:
                    logger.error("  ❌ Query is empty or None!")
                
                # Validate messages before sending
                if not messages:
                    logger.error("❌ CRITICAL: No messages to send to agent!")
                    logger.error(f"  Query: {repr(query)}")
                    logger.error(f"  System prompt exists: {hasattr(self, '_system_prompt')}")
                    logger.error(f"  History length: {len(formatted_history) if formatted_history else 0}")
                    raise ValueError("Cannot invoke agent with empty messages list")
                
                # Log message details
                msg_types = [type(m).__name__ for m in messages]
                msg_lengths = [len(m.content) if hasattr(m, 'content') else 0 for m in messages]
                logger.info(f"📤 Invoking agent with {len(messages)} messages")
                logger.debug(f"  Message types: {msg_types}")
                logger.debug(f"  Message lengths: {msg_lengths}")
                logger.debug(f"  Agent type: {getattr(self, '_agent_type', 'unknown')}")
                
                # Try messages format first (works with default create_react_agent)
                try:
                    logger.debug("  🔄 Attempting messages format invocation...")
                    
                    # Deep validation of messages
                    logger.debug(f"  Validating {len(messages)} messages before invocation...")
                    for i, msg in enumerate(messages):
                        msg_type = type(msg).__name__
                        has_content = hasattr(msg, 'content')
                        content_len = len(msg.content) if has_content else 0
                        logger.debug(f"    Message {i}: {msg_type}, has_content={has_content}, len={content_len}")
                        if not has_content or content_len == 0:
                            logger.warning(f"    ⚠️ Message {i} has no content!")
                    
                    # Ensure messages are properly formatted
                    invoke_input = {"messages": messages}
                    logger.debug(f"  Input keys: {list(invoke_input.keys())}")
                    logger.debug(f"  Messages in input: {len(invoke_input['messages'])}")
                    
                    # Serialize messages to check they're valid
                    try:
                        from langchain_core.messages import BaseMessage
                        serialized = [msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in messages]
                        logger.debug(f"  Messages can be serialized: {len(serialized)} items")
                    except Exception as ser_e:
                        logger.warning(f"  ⚠️ Message serialization check failed: {ser_e}")
                    
                    result = self.agent_executor.invoke(invoke_input, config=config)
                    
                    logger.debug(f"  ✅ Invocation successful")
                    logger.debug(f"  Result type: {type(result)}")
                    logger.debug(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    
                    # Extract output and tool calls from messages for LangGraph
                    if isinstance(result, dict):
                        if "messages" in result:
                            result_messages = result["messages"]
                            logger.debug(f"  📨 Result contains {len(result_messages)} messages")
                            
                            # Extract tool calls and results from messages (LangGraph stores them in message objects)
                            tool_calls_from_messages = []
                            tool_results_map = {}  # Map tool call IDs to results
                            
                            for msg in result_messages:
                                # Check for tool calls in AIMessage (the agent's decision to use a tool)
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get("name", "unknown")
                                        tool_args = tool_call.get("args", {})
                                        tool_id = tool_call.get("id", "")
                                        
                                        tool_calls_from_messages.append({
                                            "tool": tool_name,
                                            "input": tool_args,
                                            "id": tool_id,
                                            "result": ""  # Will be filled from ToolMessage
                                        })
                                    logger.info(f"  🔧 Found {len(msg.tool_calls)} tool call(s) in message")
                                
                                # Check for tool messages (results from tool execution)
                                # ToolMessage has 'name' attribute matching the tool name
                                if hasattr(msg, 'name') and msg.name and hasattr(msg, 'content'):
                                    # This is a tool result message
                                    tool_name = msg.name
                                    tool_result = msg.content
                                    logger.debug(f"  📦 Tool result from '{tool_name}': {len(tool_result)} chars")
                                    
                                    # Match this result to the tool call by finding the most recent matching tool call
                                    for tc in tool_calls_from_messages:
                                        if tc["tool"] == tool_name and not tc.get("result"):
                                            tc["result"] = tool_result[:3000]  # Increased limit to allow full section content
                                            break
                            
                            if result_messages:
                    # Get the last message which should be the AI response
                                last_msg = result_messages[-1]
                                logger.debug(f"  Last message type: {type(last_msg).__name__}")
                                answer = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                                
                                # Store tool calls for later extraction
                                intermediate_steps = tool_calls_from_messages if tool_calls_from_messages else []
                            else:
                                logger.warning("  ⚠️ Result messages list is empty!")
                                answer = "No response generated"
                                intermediate_steps = []
                        elif "output" in result:
                            answer = result["output"]
                            logger.debug("  ✅ Found output in result")
                            intermediate_steps = []
                        else:
                            logger.warning(f"  ⚠️ Unexpected result structure: {list(result.keys())}")
                            answer = str(result)
                            intermediate_steps = []
                    else:
                        answer = str(result)
                        logger.debug(f"  Result is not a dict: {type(result)}")
                        intermediate_steps = []
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ Messages format failed: {error_msg}")
                    logger.debug(f"  Exception type: {type(e).__name__}")
                    logger.debug(f"  Exception details: {repr(e)}", exc_info=True)
                    
                    # If messages format fails, try with input format
                    logger.info("  🔄 Trying input format as fallback...")
                    try:
                        result = self.agent_executor.invoke({"input": query}, config=config)
                        logger.debug(f"  ✅ Input format invocation successful")
                        
                        if isinstance(result, dict):
                            if "output" in result:
                                answer = result["output"]
                            elif "messages" in result:
                                result_msgs = result["messages"]
                                answer = result_msgs[-1].content if result_msgs else "No response"
                            else:
                                answer = str(result)
                        else:
                            answer = str(result)
                        intermediate_steps = []
                    except Exception as e2:
                        logger.error(f"❌ Both formats failed!")
                        logger.error(f"  Messages format error: {error_msg}")
                        logger.error(f"  Input format error: {str(e2)}")
                        raise
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
            
            # Handle LangGraph format (list of dicts) vs traditional format (list of tuples)
            if LANGCHAIN_VERSION == "1.0+" and intermediate_steps:
                # LangGraph format: intermediate_steps is already a list of tool call dicts
                for step in intermediate_steps:
                    if isinstance(step, dict):
                        tool_calls.append({
                            "tool": step.get("tool", "unknown"),
                            "input": step.get("input", {}),
                            "result": step.get("result", "")[:3000] if "result" in step else ""  # Increased limit for full content
                        })
                    else:
                        # Fallback: try to extract from tuple format
                        if len(step) >= 2:
                            tool_action = step[0]
                            tool_result = step[1]
                            tool_name = tool_action.tool if hasattr(tool_action, "tool") else "unknown"
                            tool_input = tool_action.tool_input if hasattr(tool_action, "tool_input") else {}
                            tool_calls.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": str(tool_result)[:3000]  # Increased limit for full content
                            })
            else:
                # Traditional agent format: list of (action, result) tuples
                for step in intermediate_steps:
                    if len(step) >= 2:
                        tool_action = step[0]
                        tool_result = step[1]
                    
                        tool_name = tool_action.tool if hasattr(tool_action, "tool") else "unknown"
                        tool_input = tool_action.tool_input if hasattr(tool_action, "tool_input") else {}
                        
                        tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": str(tool_result)[:3000]  # Increased limit to allow full section content
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
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.chat, query, chat_history)
    
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

