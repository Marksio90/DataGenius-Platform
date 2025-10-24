import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

@dataclass
class PerformanceMetrics:
    """Metryki wydajności systemu"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    active_sessions: int = 0
    training_queue_size: int = 0

@dataclass
class BusinessMetrics:
    """Metryki biznesowe"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_trainings: int = 0
    successful_trainings: int = 0
    avg_training_time: float = 0.0
    avg_model_accuracy: float = 0.0
    most_used_algorithms: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0

class MetricsCollector:
    """Zbieracz metryk systemowych i biznesowych"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.performance_history: deque = deque(maxlen=max_history)
        self.business_history: deque = deque(maxlen=max_history)
        self.is_collecting = False
        self.collection_thread = None
    
    def start_collection(self, interval: int = 30):
        """Rozpocznij zbieranie metryk co interval sekund"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collect_metrics_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """Zatrzymaj zbieranie metryk"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1)
    
    def _collect_metrics_loop(self, interval: int):
        """Loop zbierający metryki"""
        while self.is_collecting:
            try:
                # System metrics
                perf_metrics = PerformanceMetrics(
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_used_mb=psutil.virtual_memory().used / (1024**2),
                    disk_usage_percent=psutil.disk_usage('/').percent,
                    active_sessions=self._count_active_sessions(),
                    training_queue_size=self._get_training_queue_size()
                )
                self.performance_history.append(perf_metrics)
                
                # Business metrics
                business_metrics = self._collect_business_metrics()
                self.business_history.append(business_metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                # Log error but continue
                print(f"Metrics collection error: {e}")
                time.sleep(interval)
    
    def _count_active_sessions(self) -> int:
        """Liczba aktywnych sesji Streamlit"""
        # W rzeczywistej aplikacji - połączenie z session store
        return len([k for k in st.session_state.keys() if k.startswith('user_')])
    
    def _get_training_queue_size(self) -> int:
        """Rozmiar kolejki treningu"""
        return st.session_state.get('training_queue_size', 0)
    
    def _collect_business_metrics(self) -> BusinessMetrics:
        """Zbierz metryki biznesowe"""
        health_metrics = st.session_state.get('health_metrics', {
            'total_trainings': 0,
            'successful_trainings': 0,
            'avg_training_time': 0,
            'errors_by_type': {}
        })
        
        return BusinessMetrics(
            total_trainings=health_metrics.get('total_trainings', 0),
            successful_trainings=health_metrics.get('successful_trainings', 0),
            avg_training_time=health_metrics.get('avg_training_time', 0),
            error_rate=self._calculate_error_rate(health_metrics)
        )
    
    def _calculate_error_rate(self, health_metrics: Dict) -> float:
        """Oblicz error rate"""
        total = health_metrics.get('total_trainings', 0)
        if total == 0:
            return 0.0
        
        successful = health_metrics.get('successful_trainings', 0)
        failed = total - successful
        return (failed / total) * 100.0
    
    def get_current_performance(self) -> PerformanceMetrics:
        """Pobierz aktualne metryki wydajności"""
        if self.performance_history:
            return self.performance_history[-1]
        
        # Fallback - zbierz teraz
        return PerformanceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            memory_used_mb=psutil.virtual_memory().used / (1024**2),
            disk_usage_percent=psutil.disk_usage('/').percent
        )
    
    def get_performance_trend(self, minutes: int = 30) -> List[PerformanceMetrics]:
        """Pobierz trend wydajności z ostatnich X minut"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.performance_history if m.timestamp >= cutoff]
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Podsumowanie metryk biznesowych"""
        if not self.business_history:
            return {'status': 'no_data'}
        
        latest = self.business_history[-1]
        
        return {
            'total_trainings': latest.total_trainings,
            'success_rate': (latest.successful_trainings / max(latest.total_trainings, 1)) * 100,
            'avg_training_time': latest.avg_training_time,
            'error_rate': latest.error_rate,
            'status': 'healthy' if latest.error_rate < 10 else 'degraded' if latest.error_rate < 25 else 'unhealthy'
        }

class MonitoringDashboard:
    """Dashboard monitorowania dla adminów"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def render_admin_panel(self):
        """Renderuj panel administracyjny"""
        if not st.sidebar.checkbox("🔧 Admin Panel", key="admin_panel"):
            return
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👨‍💼 Panel Administratora")
        
        # Quick system status
        current_perf = self.metrics.get_current_performance()
        business_summary = self.metrics.get_business_summary()
        
        # Status indicator
        if business_summary['status'] == 'healthy':
            st.sidebar.success("✅ System Healthy")
        elif business_summary['status'] == 'degraded':
            st.sidebar.warning("⚠️ System Degraded")
        else:
            st.sidebar.error("🚨 System Unhealthy")
        
        # Quick metrics
        st.sidebar.metric("💻 CPU", f"{current_perf.cpu_percent:.1f}%")
        st.sidebar.metric("🧠 RAM", f"{current_perf.memory_percent:.1f}%")
        st.sidebar.metric("📊 Success Rate", f"{business_summary.get('success_rate', 0):.1f}%")
        
        # Detailed dashboard button
        if st.sidebar.button("📊 Detailed Dashboard"):
            st.session_state.show_monitoring_dashboard = True
    
    def render_detailed_dashboard(self):
        """Renderuj szczegółowy dashboard"""
        if not st.session_state.get('show_monitoring_dashboard', False):
            return
        
        st.markdown("# 📊 System Monitoring Dashboard")
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Zamknij Dashboard"):
                st.session_state.show_monitoring_dashboard = False
                st.rerun()
        
        # System Health Overview
        self._render_system_health()
        
        st.markdown("---")
        
        # Performance Charts
        self._render_performance_charts()
        
        st.markdown("---")
        
        # Business Metrics
        self._render_business_metrics()
        
        st.markdown("---")
        
        # System Information
        self._render_system_info()
    
    def _render_system_health(self):
        """Renderuj przegląd zdrowia systemu"""
        st.subheader("🏥 System Health")
        
        current_perf = self.metrics.get_current_performance()
        business_summary = self.metrics.get_business_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_color = "normal" if current_perf.cpu_percent < 70 else "inverse"
            st.metric(
                "💻 CPU Usage",
                f"{current_perf.cpu_percent:.1f}%",
                delta=None,
                delta_color=cpu_color
            )
        
        with col2:
            memory_color = "normal" if current_perf.memory_percent < 80 else "inverse"
            st.metric(
                "🧠 Memory Usage", 
                f"{current_perf.memory_percent:.1f}%",
                delta=f"{current_perf.memory_used_mb:.0f} MB",
                delta_color=memory_color
            )
        
        with col3:
            success_rate = business_summary.get('success_rate', 0)
            success_color = "normal" if success_rate > 80 else "inverse"
            st.metric(
                "✅ Success Rate",
                f"{success_rate:.1f}%",
                delta=None,
                delta_color=success_color
            )
        
        with col4:
            st.metric(
                "👥 Active Sessions",
                current_perf.active_sessions
            )
    
    def _render_performance_charts(self):
        """Renderuj wykresy wydajności"""
        st.subheader("📈 Performance Trends")
        
        # Get trend data
        trend_data = self.metrics.get_performance_trend(minutes=60)  # Last hour
        
        if not trend_data:
            st.info("📊 Brak danych trendowych - metryki będą dostępne po rozpoczęciu monitorowania")
            return
        
        # Prepare data for plotting
        timestamps = [m.timestamp for m in trend_data]
        cpu_values = [m.cpu_percent for m in trend_data]
        memory_values = [m.memory_percent for m in trend_data]
        
        # CPU and Memory trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cpu_values,
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#FF6B6B')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=memory_values,
            mode='lines+markers',
            name='Memory %',
            line=dict(color='#4ECDC4')
        ))
        
        fig.update_layout(
            title="System Resource Usage (Last Hour)",
            xaxis_title="Time",
            yaxis_title="Usage %",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_business_metrics(self):
        """Renderuj metryki biznesowe"""
        st.subheader("💼 Business Metrics")
        
        business_summary = self.metrics.get_business_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Training Statistics")
            if business_summary.get('total_trainings', 0) > 0:
                
                # Success rate pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Successful', 'Failed'],
                    values=[
                        business_summary.get('success_rate', 0),
                        100 - business_summary.get('success_rate', 0)
                    ],
                    marker_colors=['#2ECC71', '#E74C3C']
                )])
                fig_pie.update_layout(
                    title="Training Success Rate",
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No training data available yet")
        
        with col2:
            st.markdown("#### ⏱️ Performance Metrics")
            
            if business_summary.get('avg_training_time', 0) > 0:
                avg_time = business_summary['avg_training_time']
                
                # Performance indicator
                if avg_time < 30:
                    st.success(f"🚀 Fast Training: {avg_time:.1f}s average")
                elif avg_time < 120:
                    st.warning(f"⚡ Moderate Speed: {avg_time:.1f}s average") 
                else:
                    st.error(f"🐌 Slow Training: {avg_time:.1f}s average")
                
                # Training time gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_time,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Avg Training Time (s)"},
                    gauge = {
                        'axis': {'range': [None, 300]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 120], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 180
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.info("No performance data available yet")
    
    def _render_system_info(self):
        """Renderuj informacje systemowe"""
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💻 Hardware")
            st.text(f"CPU Cores: {psutil.cpu_count()}")
            st.text(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            st.text(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            
        with col2:
            st.markdown("#### 📊 Application Stats")
            st.text(f"Session State Keys: {len(st.session_state.keys())}")
            st.text(f"Cache Size: {len(st.session_state.get('smart_cache_l1', {}))}")
            
            # Uptime (approximate)
            if 'app_start_time' not in st.session_state:
                st.session_state.app_start_time = datetime.now()
            
            uptime = datetime.now() - st.session_state.app_start_time
            st.text(f"Session Uptime: {str(uptime).split('.')[0]}")

# Global metrics collector
metrics_collector = MetricsCollector()
monitoring_dashboard = MonitoringDashboard(metrics_collector)

# Start collection on import
metrics_collector.start_collection(interval=30)