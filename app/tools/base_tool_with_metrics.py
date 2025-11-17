"""
Legal Chatbot - Base Tool with Metrics Tracking
Phase 4.2: Tool Usage Statistics
"""

import time
from typing import Any, Optional
from loguru import logger
from app.core.metrics import metrics_collector

try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool


class ToolWithMetrics(BaseTool):
    """Base tool class with metrics tracking"""
    
    def _tracked_run(self, *args, **kwargs) -> str:
        """Wrapper to track tool execution time"""
        start_time = time.time()
        success = True
        result = None
        
        try:
            # Call the actual tool implementation
            result = self._run(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            logger.error(f"Tool {self.name} execution failed: {e}", exc_info=True)
            raise
        finally:
            # Record tool usage metrics
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            metrics_collector.record_tool_usage(
                tool_name=self.name,
                execution_time_ms=execution_time,
                success=success,
            )
    
    async def _tracked_arun(self, *args, **kwargs) -> str:
        """Async wrapper to track tool execution time"""
        start_time = time.time()
        success = True
        result = None
        
        try:
            # Call the actual async tool implementation
            if hasattr(self, "_arun"):
                result = await self._arun(*args, **kwargs)
            else:
                result = self._run(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            logger.error(f"Tool {self.name} async execution failed: {e}", exc_info=True)
            raise
        finally:
            # Record tool usage metrics
            execution_time = (time.time() - start_time) * 1000
            metrics_collector.record_tool_usage(
                tool_name=self.name,
                execution_time_ms=execution_time,
                success=success,
            )

