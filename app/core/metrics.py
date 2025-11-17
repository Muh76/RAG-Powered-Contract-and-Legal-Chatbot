"""
Legal Chatbot - Metrics Collection
Phase 4.2: Monitoring and Observability
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
from dataclasses import dataclass, field, asdict
from loguru import logger
import psutil
import os


@dataclass
class APIEndpointMetrics:
    """Metrics for a specific API endpoint"""
    endpoint: str
    method: str
    request_count: int = 0
    error_count: int = 0
    total_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time"""
        if self.request_count == 0:
            return 0.0
        return self.total_response_time_ms / self.request_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage"""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def record_request(self, response_time_ms: float, is_error: bool = False):
        """Record a request"""
        self.request_count += 1
        self.total_response_time_ms += response_time_ms
        self.min_response_time_ms = min(self.min_response_time_ms, response_time_ms)
        self.max_response_time_ms = max(self.max_response_time_ms, response_time_ms)
        self.recent_response_times.append(response_time_ms)
        if is_error:
            self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "min_response_time_ms": round(self.min_response_time_ms, 2) if self.min_response_time_ms != float('inf') else 0.0,
            "max_response_time_ms": round(self.max_response_time_ms, 2),
            "error_rate_percent": round(self.error_rate, 2),
        }


@dataclass
class ToolUsageStats:
    """Statistics for tool usage in agentic chat"""
    tool_name: str
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    
    def record_usage(self, execution_time_ms: float, success: bool = True):
        """Record tool usage"""
        self.usage_count += 1
        self.total_execution_time_ms += execution_time_ms
        self.avg_execution_time_ms = self.total_execution_time_ms / self.usage_count
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "success_rate_percent": round((self.success_count / self.usage_count * 100), 2) if self.usage_count > 0 else 0.0,
        }


class MetricsCollector:
    """Centralized metrics collector"""
    
    def __init__(self):
        self._lock = Lock()
        self._endpoint_metrics: Dict[str, APIEndpointMetrics] = {}
        self._tool_usage_stats: Dict[str, ToolUsageStats] = {}
        self._start_time = datetime.utcnow()
    
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
    ):
        """Record an API request"""
        key = f"{method}:{endpoint}"
        is_error = status_code >= 400
        
        with self._lock:
            if key not in self._endpoint_metrics:
                self._endpoint_metrics[key] = APIEndpointMetrics(
                    endpoint=endpoint,
                    method=method,
                )
            self._endpoint_metrics[key].record_request(response_time_ms, is_error)
        
        # Log metrics
        logger.debug(
            "API request recorded",
            extra={
                "endpoint": endpoint,
                "method": method,
                "response_time_ms": round(response_time_ms, 2),
                "status_code": status_code,
                "is_error": is_error,
                "type": "metrics",
            },
        )
    
    def record_tool_usage(
        self,
        tool_name: str,
        execution_time_ms: float,
        success: bool = True,
    ):
        """Record tool usage in agentic chat"""
        with self._lock:
            if tool_name not in self._tool_usage_stats:
                self._tool_usage_stats[tool_name] = ToolUsageStats(tool_name=tool_name)
            self._tool_usage_stats[tool_name].record_usage(execution_time_ms, success)
        
        # Log tool usage
        logger.info(
            "Tool usage recorded",
            extra={
                "tool_name": tool_name,
                "execution_time_ms": round(execution_time_ms, 2),
                "success": success,
                "type": "tool_usage",
            },
        )
    
    def get_endpoint_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for endpoint(s)"""
        with self._lock:
            if endpoint:
                key = endpoint
                if key in self._endpoint_metrics:
                    return self._endpoint_metrics[key].to_dict()
                return {}
            else:
                return {
                    key: metrics.to_dict()
                    for key, metrics in self._endpoint_metrics.items()
                }
    
    def get_tool_usage_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get tool usage statistics"""
        with self._lock:
            if tool_name:
                if tool_name in self._tool_usage_stats:
                    return self._tool_usage_stats[tool_name].to_dict()
                return {}
            else:
                return {
                    name: stats.to_dict()
                    for name, stats in self._tool_usage_stats.items()
                }
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics"""
        with self._lock:
            total_requests = sum(m.request_count for m in self._endpoint_metrics.values())
            total_errors = sum(m.error_count for m in self._endpoint_metrics.values())
            
            all_response_times = []
            for metrics in self._endpoint_metrics.values():
                all_response_times.extend(metrics.recent_response_times)
            
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0.0
            
            total_tool_usage = sum(stats.usage_count for stats in self._tool_usage_stats.values())
            
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_error_rate_percent": round((total_errors / total_requests * 100), 2) if total_requests > 0 else 0.0,
                "avg_response_time_ms": round(avg_response_time, 2),
                "total_tool_usage": total_tool_usage,
                "endpoints_tracked": len(self._endpoint_metrics),
                "tools_tracked": len(self._tool_usage_stats),
            }
    
    def reset_metrics(self):
        """Reset all metrics (for testing)"""
        with self._lock:
            self._endpoint_metrics.clear()
            self._tool_usage_stats.clear()
            self._start_time = datetime.utcnow()


class SystemMetrics:
    """System metrics collector (CPU, memory, disk)"""
    
    @staticmethod
    def get_cpu_metrics() -> Dict[str, Any]:
        """Get CPU metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            return {
                "cpu_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "cpu_freq_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
            }
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
            return {"cpu_percent": None, "cpu_count": None, "cpu_freq_mhz": None}
    
    @staticmethod
    def get_memory_metrics() -> Dict[str, Any]:
        """Get memory metrics"""
        try:
            mem = psutil.virtual_memory()
            return {
                "memory_total_gb": round(mem.total / (1024**3), 2),
                "memory_available_gb": round(mem.available / (1024**3), 2),
                "memory_used_gb": round(mem.used / (1024**3), 2),
                "memory_percent": round(mem.percent, 2),
            }
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
            return {
                "memory_total_gb": None,
                "memory_available_gb": None,
                "memory_used_gb": None,
                "memory_percent": None,
            }
    
    @staticmethod
    def get_disk_metrics() -> Dict[str, Any]:
        """Get disk metrics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": round(disk.percent, 2),
            }
        except Exception as e:
            logger.warning(f"Failed to get disk metrics: {e}")
            return {
                "disk_total_gb": None,
                "disk_used_gb": None,
                "disk_free_gb": None,
                "disk_percent": None,
            }
    
    @staticmethod
    def get_all_metrics() -> Dict[str, Any]:
        """Get all system metrics"""
        return {
            "cpu": SystemMetrics.get_cpu_metrics(),
            "memory": SystemMetrics.get_memory_metrics(),
            "disk": SystemMetrics.get_disk_metrics(),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()

