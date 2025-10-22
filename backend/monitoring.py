from __future__ import annotations

# backend/monitoring.py

from backend.safe_utils import truthy_df_safe

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

# --- Opcjonalne zależności (bezpieczne importy) ---
try:
    import psutil  # type: ignore
except Exception:  # fallback gdy brak psutil
    psutil = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # type: ignore


# ===================== POMOCNICZE =====================

def _has_st() -> bool:
    return st is not None

def _has_session_state() -> bool:
    return _has_st() and hasattr(st, "session_state")

def _truthy(x: Any) -> bool:
    try:
        return bool(truthy_df_safe(x))
    except Exception:
        return bool(x)


# ===================== MODELE DANYCH =====================

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


# ===================== KOLLEKTOR METRYK =====================

class MetricsCollector:
    """Zbieracz metryk systemowych i biznesowych"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.performance_history: deque[PerformanceMetrics] = deque(maxlen=max_history)
        self.business_history: deque[BusinessMetrics] = deque(maxlen=max_history)
        self.is_collecting: bool = False
        self.collection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _ensure_collection_thread(self, interval: int = 30):
        """Upewnij się, że jest tylko jeden wątek zbierający metryki."""
        with self._lock:
            if self.is_collecting:
                return
            if self.collection_thread and self.collection_thread.is_alive():
                # Wątek już aktywny
                self.is_collecting = True
                return

            # Start collection
            self.is_collecting = True
            self.collection_thread = threading.Thread(
                target=self._collect_metrics_loop, args=(interval,), daemon=True
            )
            self.collection_thread.start()

    def start_collection(self, interval: int = 30):
        """Rozpocznij zbieranie metryk co `interval` sekund."""
        self._ensure_collection_thread(interval=interval)

    def stop_collection(self):
        """Zatrzymaj zbieranie metryk."""
        with self._lock:
            self.is_collecting = False
        if self.collection_thread:
            try:
                self.collection_thread.join(timeout=1.0)
            except Exception:
                pass

    def _collect_metrics_loop(self, interval: int):
        """Pętla zbierająca metryki."""
        while self.is_collecting:
            try:
                perf_metrics = self._collect_performance_metrics()
                self.performance_history.append(perf_metrics)
            except Exception as e:
                print(f"[metrics] performance collection error: {e}")

            try:
                business_metrics = self._collect_business_metrics()
                self.business_history.append(business_metrics)
            except Exception as e:
                print(f"[metrics] business collection error: {e}")

            time.sleep(max(1, int(interval)))

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_mb = 0.0
        disk_usage_percent = 0.0

        if psutil is not None:
            try:
                # cpu_percent z interwałem minimalnym – nie blokuj długim pomiarem
                cpu_percent = float(psutil.cpu_percent(interval=0.2))
                vm = psutil.virtual_memory()
                memory_percent = float(vm.percent)
                memory_used_mb = float(vm.used) / (1024 ** 2)
                disk_usage_percent = float(psutil.disk_usage('/').percent)
            except Exception:
                pass

        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_usage_percent=disk_usage_percent,
            active_sessions=self._count_active_sessions(),
            training_queue_size=self._get_training_queue_size(),
        )

    def _count_active_sessions(self) -> int:
        """Liczba aktywnych sesji (heurystyka po kluczach)."""
        if not _has_session_state():
            return 0
        try:
            return len([k for k in st.session_state.keys() if str(k).startswith('user_')])
        except Exception:
            return 0

    def _get_training_queue_size(self) -> int:
        """Rozmiar kolejki treningu."""
        if not _has_session_state():
            return 0
        try:
            return int(st.session_state.get('training_queue_size', 0) or 0)
        except Exception:
            return 0

    def _collect_business_metrics(self) -> BusinessMetrics:
        """Zbierz metryki biznesowe z session_state."""
        default_health = {
            'total_trainings': 0,
            'successful_trainings': 0,
            'avg_training_time': 0.0,
            'errors_by_type': {}
        }
        health_metrics = default_health
        if _has_session_state():
            try:
                health_metrics = st.session_state.get('health_metrics', default_health) or default_health
            except Exception:
                pass

        return BusinessMetrics(
            total_trainings=int(health_metrics.get('total_trainings', 0) or 0),
            successful_trainings=int(health_metrics.get('successful_trainings', 0) or 0),
            avg_training_time=float(health_metrics.get('avg_training_time', 0) or 0.0),
            error_rate=self._calculate_error_rate(health_metrics),
        )

    def _calculate_error_rate(self, health_metrics: Dict[str, Any]) -> float:
        try:
            total = int(health_metrics.get('total_trainings', 0) or 0)
            if total <= 0:
                return 0.0
            successful = int(health_metrics.get('successful_trainings', 0) or 0)
            failed = max(0, total - successful)
            return (failed / total) * 100.0
        except Exception:
            return 0.0

    def get_current_performance(self) -> PerformanceMetrics:
        if self.performance_history:
            return self.performance_history[-1]
        # fallback natychmiastowy snapshot
        return self._collect_performance_metrics()

    def get_performance_trend(self, minutes: int = 30) -> List[PerformanceMetrics]:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.performance_history if m.timestamp >= cutoff]

    def get_business_summary(self) -> Dict[str, Any]:
        if not self.business_history:
            return {
                'status': 'no_data',
                'success_rate': 0.0,
                'total_trainings': 0,
                'avg_training_time': 0.0,
                'error_rate': 0.0
            }
        latest = self.business_history[-1]
        success_rate = (latest.successful_trainings / max(latest.total_trainings, 1)) * 100.0
        status = 'healthy' if latest.error_rate < 10 else ('degraded' if latest.error_rate < 25 else 'unhealthy')
        return {
            'total_trainings': latest.total_trainings,
            'success_rate': success_rate,
            'avg_training_time': latest.avg_training_time,
            'error_rate': latest.error_rate,
            'status': status,
        }


# ===================== DASHBOARD =====================

class MonitoringDashboard:
    """Dashboard monitorowania dla adminów"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

    # ✅ Wersja BEZ expandera – sama zawartość panelu
    def render_admin_panel(self, *, parent=None):
        """
        Renderuje treść Panelu Administratora bez tworzenia expandera.
        Używaj wewnątrz własnego st.expander(...) lub bezpośrednio.
        """
        if not _has_st():
            return
        root = parent if parent is not None else st

        root.markdown("### 👨‍💼 Panel Administratora")

        current_perf = self.metrics.get_current_performance()
        business_summary = self.metrics.get_business_summary()

        # Status
        status = str(business_summary.get('status', 'no_data')).lower()
        if status == 'healthy':
            root.success("✅ System Healthy")
        elif status == 'degraded':
            root.warning("⚠️ System Degraded")
        elif status == 'unhealthy':
            root.error("🚨 System Unhealthy")
        else:
            root.info("ℹ️ Brak danych")

        # Szybkie metryki
        c1, c2, c3 = root.columns(3)
        with c1:
            root.metric("💻 CPU", f"{current_perf.cpu_percent:.1f}%")
        with c2:
            root.metric("🧠 RAM", f"{current_perf.memory_percent:.1f}%")
        with c3:
            sr = float(business_summary.get('success_rate', 0.0) or 0.0)
            root.metric("📊 Success Rate", f"{sr:.1f}%")

        # Przejście do pełnego dashboardu
        if root.button("📊 Detailed Dashboard", key="btn_detailed_dashboard"):
            if _has_session_state():
                st.session_state.show_monitoring_dashboard = True

    # ✅ Wygodny wrapper – tworzy expander i woła panel
    def render_admin_expander(self, *, parent=None, expanded: bool = False, title: str = "🛠️ Admin Panel"):
        if not _has_st():
            return
        root = parent if parent is not None else st
        with root.expander(title, expanded=expanded):
            self.render_admin_panel(parent=root)

    def render_detailed_dashboard(self):
        """Renderuje szczegółowy dashboard (pełny widok)."""
        if not _has_st() or not _has_session_state():
            return
        if not st.session_state.get('show_monitoring_dashboard', False):
            return

        st.markdown("# 📊 System Monitoring Dashboard")

        # Close button
        _, _, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Zamknij Dashboard", key="btn_close_dashboard"):
                st.session_state.show_monitoring_dashboard = False
                st.rerun()

        self._render_system_health()
        st.markdown("---")
        self._render_performance_charts()
        st.markdown("---")
        self._render_business_metrics()
        st.markdown("---")
        self._render_system_info()

    def _render_system_health(self):
        if not _has_st():
            return
        st.subheader("🏥 System Health")

        current_perf = self.metrics.get_current_performance()
        business_summary = self.metrics.get_business_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cpu_color = "normal" if current_perf.cpu_percent < 70 else "inverse"
            st.metric("💻 CPU Usage", f"{current_perf.cpu_percent:.1f}%", delta=None, delta_color=cpu_color)
        with col2:
            memory_color = "normal" if current_perf.memory_percent < 80 else "inverse"
            st.metric(
                "🧠 Memory Usage",
                f"{current_perf.memory_percent:.1f}%",
                delta=f"{current_perf.memory_used_mb:.0f} MB",
                delta_color=memory_color
            )
        with col3:
            success_rate = float(business_summary.get('success_rate', 0.0) or 0.0)
            success_color = "normal" if success_rate > 80 else "inverse"
            st.metric("✅ Success Rate", f"{success_rate:.1f}%", delta=None, delta_color=success_color)
        with col4:
            st.metric("👥 Active Sessions", current_perf.active_sessions)

    def _render_performance_charts(self):
        if not _has_st():
            return
        st.subheader("📈 Performance Trends")

        trend = self.metrics.get_performance_trend(minutes=60)
        if not _truthy(trend):
            st.info("📊 Brak danych trendowych – zacznij monitorowanie i poczekaj na pierwsze próbki.")
            return

        if go is None:
            st.info("Plotly niedostępne – zainstaluj `plotly`, aby zobaczyć wykresy.")
            return

        timestamps = [m.timestamp for m in trend]
        cpu_values = [m.cpu_percent for m in trend]
        memory_values = [m.memory_percent for m in trend]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=cpu_values, mode='lines+markers', name='CPU %'))
        fig.add_trace(go.Scatter(x=timestamps, y=memory_values, mode='lines+markers', name='Memory %'))
        fig.update_layout(
            title="System Resource Usage (Last Hour)",
            xaxis_title="Time",
            yaxis_title="Usage %",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_business_metrics(self):
        if not _has_st():
            return
        st.subheader("💼 Business Metrics")
        bs = self.metrics.get_business_summary()

        if go is None:
            st.info("Plotly niedostępne – zainstaluj `plotly`, aby zobaczyć wykresy.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Training Statistics")
            total = int(bs.get('total_trainings', 0) or 0)
            if total > 0:
                ok_pct = float(bs.get('success_rate', 0.0) or 0.0)
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Successful', 'Failed'],
                    values=[ok_pct, max(0.0, 100.0 - ok_pct)]
                )])
                fig_pie.update_layout(title="Training Success Rate", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No training data available yet")

        with col2:
            st.markdown("#### ⏱️ Performance Metrics")
            avg_time = float(bs.get('avg_training_time', 0.0) or 0.0)
            if avg_time > 0:
                # Wskaźnik (gauge)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_time,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Avg Training Time (s)"},
                    gauge={
                        'axis': {'range': [None, 300]},
                        # kolory pozostawione domyślne – zgodnie z zaleceniami stylów
                        'steps': [{'range': [0, 30]}, {'range': [30, 120]}],
                        'threshold': {'value': 180},
                    }
                ))
                fig_g.update_layout(height=300)
                # Wskazówka tekstowa
                if avg_time < 30:
                    st.success(f"🚀 Fast Training: {avg_time:.1f}s average")
                elif avg_time < 120:
                    st.warning(f"⚡ Moderate Speed: {avg_time:.1f}s average")
                else:
                    st.error(f"🐌 Slow Training: {avg_time:.1f}s average")
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("No performance data available yet")

    def _render_system_info(self):
        if not _has_st():
            return
        st.subheader("🖥️ System Information")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💻 Hardware")
            if psutil is not None:
                try:
                    st.text(f"CPU Cores: {psutil.cpu_count()}")
                    vm = psutil.virtual_memory()
                    st.text(f"Total RAM: {vm.total / (1024**3):.1f} GB")
                    st.text(f"Available RAM: {vm.available / (1024**3):.1f} GB")
                except Exception:
                    st.text("Hardware info unavailable")
            else:
                st.text("psutil not installed – hardware info unavailable")

        with col2:
            st.markdown("#### 📊 Application Stats")
            try:
                ss_len = len(st.session_state.keys()) if _has_session_state() else 0
                cache_len = len(st.session_state.get('smart_cache_l1', {})) if _has_session_state() else 0
                st.text(f"Session State Keys: {ss_len}")
                st.text(f"Cache Size: {cache_len}")
                if _has_session_state():
                    if 'app_start_time' not in st.session_state:
                        st.session_state.app_start_time = datetime.now()
                    uptime = datetime.now() - st.session_state.app_start_time
                    st.text(f"Session Uptime: {str(uptime).split('.')[0]}")
            except Exception:
                st.text("App stats unavailable")


# ===================== SINGLETONY =====================

# Bezpieczna inicjalizacja tylko gdy Streamlit i session_state dostępne
if _has_session_state():
    if "metrics_collector" not in st.session_state:
        collector = MetricsCollector()
        collector.start_collection(interval=30)
        st.session_state.metrics_collector = collector
    elif not st.session_state.metrics_collector.is_collecting:
        st.session_state.metrics_collector.start_collection(interval=30)

    metrics_collector: MetricsCollector = st.session_state.metrics_collector
    monitoring_dashboard = MonitoringDashboard(metrics_collector)
else:
    # Fallback poza Streamlitem – wciąż pozwala importować moduł
    metrics_collector = MetricsCollector()
    monitoring_dashboard = MonitoringDashboard(metrics_collector)
