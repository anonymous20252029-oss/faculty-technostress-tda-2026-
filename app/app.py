import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from utils import load_or_generate_baseline, project_out_of_sample, save_pilot_response

st.set_page_config(
    page_title="TDA Cognitive Load & Technostress Benchmarking",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom responsive CSS optimized for mobile viewports
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3rem; }
    .metric-card { background-color: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# Parse group ID from URL query parameters (e.g., ?group=CNTT_TDTU)
query_params = st.query_params
group_id = query_params.get("group", "Conference_Live")

st.title("🧠 Cognitive Friction & Technostress Benchmarking")
st.markdown(f"**In-Situ Micro-Survey Engine** | **Cohort Identifier:** `{group_id}`")

# Initialize unobtrusive client-side latency timer
if "init_time" not in st.session_state:
    st.session_state.init_time = time.time()

# Load global empirical baseline (N = 3,459) and trained PCA projection
df_base, X_base, pca, X_pca_base = load_or_generate_baseline()

col_survey, col_vis = st.columns([1, 1.2])

with col_survey:
    st.subheader("1. In-Situ Micro-Assessment (3 Targeted Items)")
    with st.form("micro_survey_form"):
        q1 = st.slider(
            "1. Post-work affective exhaustion and energy depletion (F1 - Burnout Severity):",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5,
            help="1: Completely energized, 5: Total cognitive and physical exhaustion"
        )
        q2 = st.slider(
            "2. Stress and cognitive friction from continuous AI tool adoption (F2 - AI Technostress):",
            min_value=1.0, max_value=5.0, value=3.5, step=0.5,
            help="1: Seamless effortless workflow, 5: Chronic digital overload & interface churn"
        )
        q3 = st.slider(
            "3. Decision hesitation and coping delay when multitasking (F3 - Latency Proxy):",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5,
            help="1: Rapid execution, 5: Frequent hesitation and procedural friction"
        )
        
        submitted = st.form_submit_button("🚀 Submit & Project onto Global Manifold")

    if submitted:
        total_latency = round(time.time() - st.session_state.init_time, 2)
        st.session_state.user_vector = [q1, q2, q3]
        st.session_state.total_latency = total_latency

        # Record anonymous response
        record = {
            "Burnout_Level": q1,
            "AI_Technostress": q2,
            "Cognitive_Latency_Proxy": q3,
            "group_id": group_id,
            "total_latency_sec": total_latency,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        save_pilot_response(record)
        st.success(f"✅ Projection successful! Recorded decision latency: {total_latency} s.")

with col_vis:
    st.subheader("2. Out-of-Sample Manifold Projection")

    fig = go.Figure()

    # Layer 1: Global Empirical Normative Baseline (N = 3,459)
    fig.add_trace(go.Scattergl(
        x=X_pca_base[:, 0],
        y=X_pca_base[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color=X_base[:, 1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Technostress (F2)", len=0.7)
        ),
        name="Global Normative Baseline (N = 3,459)",
        opacity=0.35
    ))

    # Layer 2: Incoming Individual Out-of-Sample Vector
    if "user_vector" in st.session_state:
        u_coords = project_out_of_sample(st.session_state.user_vector, pca)
        fig.add_trace(go.Scatter(
            x=[u_coords[0]],
            y=[u_coords[1]],
            mode='markers+text',
            marker=dict(size=18, color='#e11d48', symbol='star', line=dict(color='white', width=2)),
            text=["Your Coordinates"],
            textposition="top center",
            name="Your Current State"
        ))

    fig.update_layout(
        xaxis_title=f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)",
        yaxis_title=f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

# Three-tier diagnostic feedback & policy recommendations
if "user_vector" in st.session_state:
    u = st.session_state.user_vector
    st.divider()
    st.subheader("💡 Diagnostic Feedback & Evidence-Based Recommendations")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Burnout Severity (F1)", f"{u[0]:.1f} / 5.0", delta=f"{u[0] - 3.09:+.2f} vs Norms")
    with col_b:
        st.metric("AI Technostress (F2)", f"{u[1]:.1f} / 5.0", delta=f"{u[1] - 3.38:+.2f} vs Norms")
    with col_c:
        st.metric("Cognitive Latency (F3)", f"{u[2]:.1f} / 5.0", delta=f"{u[2] - 2.86:+.2f} vs Norms")

    if u[1] >= 4.0:
        st.warning(
            "⚠️ **Elevated Technostress Alert:** Your latent coordinate is located in the upper hemisphere, "
            "structurally adjacent to enterprise software engineering stress regimes. "
            "**Recommended Action:** Establish strict asynchronous digital boundaries, integrate structured micro-breaks, "
            "and advocate for phased institutional rollout schedules for new generative AI platforms."
        )
    else:
        st.info(
            "🌱 **Adaptive Equilibrium:** Your technological workload and cognitive coping parameters are "
            "currently situated within a stable, sustainable operational zone."
        )
