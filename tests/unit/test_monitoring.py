"""
Unit tests for Phase 4.2: Monitoring and Observability
"""

import pytest
import time
from datetime import datetime
from app.core.metrics import metrics_collector, SystemMetrics, APIEndpointMetrics, ToolUsageStats
from app.core.health_checker import health_checker
from app.core.logging import setup_logging


class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector can be initialized"""
        assert metrics_collector is not None
        assert hasattr(metrics_collector, "record_api_request")
        assert hasattr(metrics_collector, "record_tool_usage")
        assert hasattr(metrics_collector, "get_summary_metrics")
    
    def test_api_request_recording(self):
        """Test API request metrics recording"""
        metrics_collector.reset_metrics()
        
        metrics_collector.record_api_request(
            endpoint="/api/v1/test",
            method="GET",
            response_time_ms=100.5,
            status_code=200,
        )
        
        summary = metrics_collector.get_summary_metrics()
        assert summary["total_requests"] == 1
        assert summary["total_errors"] == 0
    
    def test_api_error_recording(self):
        """Test API error metrics recording"""
        metrics_collector.reset_metrics()
        
        metrics_collector.record_api_request(
            endpoint="/api/v1/test",
            method="GET",
            response_time_ms=50.0,
            status_code=500,
        )
        
        summary = metrics_collector.get_summary_metrics()
        assert summary["total_requests"] == 1
        assert summary["total_errors"] == 1
    
    def test_tool_usage_recording(self):
        """Test tool usage metrics recording"""
        metrics_collector.reset_metrics()
        
        metrics_collector.record_tool_usage(
            tool_name="test_tool",
            execution_time_ms=75.0,
            success=True,
        )
        
        tool_stats = metrics_collector.get_tool_usage_stats()
        assert "test_tool" in tool_stats
        assert tool_stats["test_tool"]["usage_count"] == 1
        assert tool_stats["test_tool"]["success_count"] == 1
    
    def test_tool_failure_recording(self):
        """Test tool failure metrics recording"""
        metrics_collector.reset_metrics()
        
        metrics_collector.record_tool_usage(
            tool_name="test_tool",
            execution_time_ms=50.0,
            success=False,
        )
        
        tool_stats = metrics_collector.get_tool_usage_stats()
        assert tool_stats["test_tool"]["usage_count"] == 1
        assert tool_stats["test_tool"]["failure_count"] == 1
        assert tool_stats["test_tool"]["success_count"] == 0
    
    def test_endpoint_metrics_calculation(self):
        """Test endpoint metrics calculations"""
        metrics_collector.reset_metrics()
        
        # Record multiple requests
        for i in range(5):
            metrics_collector.record_api_request(
                endpoint="/api/v1/test",
                method="GET",
                response_time_ms=100.0 + (i * 10),
                status_code=200 if i < 4 else 500,
            )
        
        endpoint_metrics = metrics_collector.get_endpoint_metrics("GET:/api/v1/test")
        assert endpoint_metrics["request_count"] == 5
        assert endpoint_metrics["error_count"] == 1
        assert endpoint_metrics["error_rate_percent"] == 20.0
        assert endpoint_metrics["avg_response_time_ms"] == 120.0
    
    def test_summary_metrics(self):
        """Test summary metrics calculation"""
        metrics_collector.reset_metrics()
        
        # Record some requests and tool usage
        metrics_collector.record_api_request("/api/v1/test", "GET", 100.0, 200)
        metrics_collector.record_tool_usage("test_tool", 50.0, True)
        
        summary = metrics_collector.get_summary_metrics()
        assert summary["total_requests"] == 1
        assert summary["total_tool_usage"] == 1
        assert "uptime_seconds" in summary


class TestSystemMetrics:
    """Test system metrics collection"""
    
    def test_cpu_metrics(self):
        """Test CPU metrics collection"""
        cpu_metrics = SystemMetrics.get_cpu_metrics()
        assert "cpu_percent" in cpu_metrics
        assert "cpu_count" in cpu_metrics
        # CPU percent should be 0-100 or None
        if cpu_metrics["cpu_percent"] is not None:
            assert 0 <= cpu_metrics["cpu_percent"] <= 100
    
    def test_memory_metrics(self):
        """Test memory metrics collection"""
        memory_metrics = SystemMetrics.get_memory_metrics()
        assert "memory_total_gb" in memory_metrics
        assert "memory_available_gb" in memory_metrics
        assert "memory_used_gb" in memory_metrics
        assert "memory_percent" in memory_metrics
        # Memory percent should be 0-100 or None
        if memory_metrics["memory_percent"] is not None:
            assert 0 <= memory_metrics["memory_percent"] <= 100
    
    def test_disk_metrics(self):
        """Test disk metrics collection"""
        disk_metrics = SystemMetrics.get_disk_metrics()
        assert "disk_total_gb" in disk_metrics
        assert "disk_used_gb" in disk_metrics
        assert "disk_free_gb" in disk_metrics
        assert "disk_percent" in disk_metrics
        # Disk percent should be 0-100 or None
        if disk_metrics["disk_percent"] is not None:
            assert 0 <= disk_metrics["disk_percent"] <= 100
    
    def test_all_metrics(self):
        """Test all system metrics collection"""
        all_metrics = SystemMetrics.get_all_metrics()
        assert "cpu" in all_metrics
        assert "memory" in all_metrics
        assert "disk" in all_metrics
        assert "timestamp" in all_metrics


class TestHealthChecker:
    """Test health checker functionality"""
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self):
        """Test health checker can be initialized"""
        assert health_checker is not None
        assert hasattr(health_checker, "check_all_dependencies")
    
    @pytest.mark.asyncio
    async def test_all_dependencies_check(self):
        """Test checking all dependencies"""
        dependencies = await health_checker.check_all_dependencies()
        assert isinstance(dependencies, dict)
        assert "database" in dependencies
        assert "redis" in dependencies
        assert "vector_store" in dependencies
        assert "llm_api" in dependencies
    
    @pytest.mark.asyncio
    async def test_dependency_status_format(self):
        """Test dependency status format"""
        dependencies = await health_checker.check_all_dependencies()
        for service, status in dependencies.items():
            assert "status" in status
            assert status["status"] in ["healthy", "unhealthy", "unknown", "error"]


class TestLogging:
    """Test logging functionality"""
    
    def test_logging_setup(self):
        """Test logging can be set up"""
        logger = setup_logging()
        assert logger is not None
    
    def test_logging_levels(self):
        """Test logging levels"""
        logger = setup_logging()
        # Test that logger has common methods
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")


class TestAPIEndpointMetrics:
    """Test APIEndpointMetrics class"""
    
    def test_endpoint_metrics_creation(self):
        """Test creating endpoint metrics"""
        metrics = APIEndpointMetrics(endpoint="/api/v1/test", method="GET")
        assert metrics.endpoint == "/api/v1/test"
        assert metrics.method == "GET"
        assert metrics.request_count == 0
    
    def test_endpoint_metrics_recording(self):
        """Test recording endpoint metrics"""
        metrics = APIEndpointMetrics(endpoint="/api/v1/test", method="GET")
        metrics.record_request(100.0, is_error=False)
        metrics.record_request(200.0, is_error=False)
        metrics.record_request(150.0, is_error=True)
        
        assert metrics.request_count == 3
        assert metrics.error_count == 1
        assert metrics.avg_response_time_ms == 150.0
        assert metrics.error_rate == pytest.approx(33.33, rel=0.1)


class TestToolUsageStats:
    """Test ToolUsageStats class"""
    
    def test_tool_stats_creation(self):
        """Test creating tool usage stats"""
        stats = ToolUsageStats(tool_name="test_tool")
        assert stats.tool_name == "test_tool"
        assert stats.usage_count == 0
    
    def test_tool_stats_recording(self):
        """Test recording tool usage"""
        stats = ToolUsageStats(tool_name="test_tool")
        stats.record_usage(100.0, success=True)
        stats.record_usage(150.0, success=True)
        stats.record_usage(50.0, success=False)
        
        assert stats.usage_count == 3
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.avg_execution_time_ms == pytest.approx(100.0, rel=0.1)

