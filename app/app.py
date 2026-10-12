import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import io
import json
import base64
import requests
import urllib.parse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import qrcode

# ==============================================================================
# 1. COMPACT VIEWPORT CONFIGURATION (NO VERTICAL SCROLLING)
# ==============================================================================
st.set_page_config(
    page_title="Cognitive Technostress Benchmarking",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 1100px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h3, h4, h5 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }
    p, label {
        font-size: 0.9rem !important;
    }
    .stRadio > div {
        gap: 0.25rem !important;
    }
    .compact-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 10px 12px;
        height: 100%;
        border: 1px solid #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .group-badge {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Định vị đường dẫn dữ liệu chuẩn cấu trúc GitHub Repo
DATA_GLOBAL_PATH = "data/processed/global_aligned_real_dataset.csv"
DATA_PILOT_PATH = "data/pilot/pilot_survey_cntt_30_responses.csv"
BASE_URL = "https://faculty-technostress-tda-mmmrqgaetftbvoqpdigqmm.streamlit.app"

url_group = st.query_params.get("group", None)
if url_group:
    url_group = urllib.parse.unquote(url_group).strip()

url_page = st.query_params.get("view", "survey")

if 'dt_start' not in st.session_state:
    st.session_state.dt_start = datetime.now()
    st.session_state.t0 = time.time()
if 't_q1' not in st.session_state:
    st.session_state.t_q1 = None
if 't_q2' not in st.session_state:
    st.session_state.t_q2 = None
if 't_q3' not in st.session_state:
    st.session_state.t_q3 = None
if 'latest_user' not in st.session_state:
    st.session_state.latest_user = None

def on_select_q1():
    st.session_state.t_q1 = time.time()

def on_select_q2():
    st.session_state.t_q2 = time.time()

def on_select_q3():
    st.session_state.t_q3 = time.time()

# ==============================================================================
# 2. GITHUB REST API ENGINE: DUAL STORAGE (LOCAL & REMOTE)
# ==============================================================================
def get_github_data():
    """Retrieve existing survey responses from GitHub repository or local fallback."""
    if "github" not in st.secrets:
        if os.path.exists(DATA_PILOT_PATH):
            return pd.read_csv(DATA_PILOT_PATH), None
        return pd.DataFrame(), None

    gh = st.secrets["github"]
    # gh['file_path'] trỏ vào: data/pilot/pilot_survey_cntt_30_responses.csv
    file_path = gh.get("file_path", DATA_PILOT_PATH)
    url = f"https://api.github.com/repos/{gh['repo']}/contents/{file_path}?ref={gh.get('branch', 'main')}"
    headers = {
        "Authorization": f"Bearer {gh['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            content_json = res.json()
            sha = content_json.get("sha")
            csv_content = base64.b64decode(content_json["content"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_content))
            return df, sha
    except Exception:
        pass
    if os.path.exists(DATA_PILOT_PATH):
        return pd.read_csv(DATA_PILOT_PATH), None
    return pd.DataFrame(), None

def commit_to_github(new_record):
    """Commit anonymous survey responses to GitHub repo via REST API or local storage."""
    df, sha = get_github_data()
    df_new = pd.DataFrame([new_record])
    df_combined = df_new if df.empty else pd.concat([df, df_new], ignore_index=True)
    csv_str = df_combined.to_csv(index=False)
    content_b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
    
    if "github" in st.secrets:
        gh = st.secrets["github"]
        file_path = gh.get("file_path", DATA_PILOT_PATH)
        url = f"https://api.github.com/repos/{gh['repo']}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {gh['token']}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "message": f"feat: in-situ micro-survey entry group {new_record['group_id']}",
            "content": content_b64,
            "branch": gh.get("branch", "main")
        }
        if sha:
            payload["sha"] = sha
        res = requests.put(url, headers=headers, json=payload)
        return res.status_code in [200, 201], df_combined
    else:
        os.makedirs(os.path.dirname(DATA_PILOT_PATH), exist_ok=True)
        df_combined.to_csv(DATA_PILOT_PATH, index=False)
        return True, df_combined

# ==============================================================================
# 3. GLOBAL BASELINE FIT (N=3,459) & PCA REDUCTION
# ==============================================================================
@st.cache_resource
def load_and_fit_pca():
    if os.path.exists(DATA_GLOBAL_PATH):
        df_global = pd.read_csv(DATA_GLOBAL_PATH)
    else:
        # Fallback generator honoring baseline distribution
        np.random.seed(42)
        df_global = pd.DataFrame({
            'Burnout_Level': np.clip(np.random.normal(3.09, 0.85, 3459), 1.0, 5.0),
            'AI_Technostress': np.clip(np.random.normal(3.38, 0.95, 3459), 1.0, 5.0),
            'Cognitive_Latency_Proxy': np.clip(np.random.normal(2.86, 0.75, 3459), 1.0, 5.0)
        })
    feature_cols = ['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_global[feature_cols].values)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    df_global['PC1'] = X_2d[:, 0]
    df_global['PC2'] = X_2d[:, 1]
    
    df_bg_sample = df_global.sample(n=min(500, len(df_global)), random_state=42)
    return df_global, df_bg_sample, scaler, pca

df_global, df_bg_sample, scaler, pca = load_and_fit_pca()

def generate_qr_image(link_url):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(link_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ==============================================================================
# 4. SIDEBAR: COHORT QR-CODE & LINK GENERATOR
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔗 Cohort Routing & QR Generator")
    new_group_name = st.text_input("Create Cohort ID:", placeholder="e.g., CNTT_TDTU, Pilot_30")
    if new_group_name:
        encoded_grp = urllib.parse.quote(new_group_name.strip())
        full_group_url = f"{BASE_URL}/?group={encoded_grp}"
        st.success(f"Cohort Link: {new_group_name}")
        st.code(full_group_url, language="text")
        qr_bytes = generate_qr_image(full_group_url)
        st.image(qr_bytes, caption=f"QR Code: {new_group_name}", use_container_width=True)
        st.download_button("📥 Download QR Code", qr_bytes, file_name=f"QR_{new_group_name}.png", mime="image/png")

# ==============================================================================
# VIEW 1: IN-SITU MICRO-ASSESSMENT FORM
# ==============================================================================
if url_page != "result":
    st.markdown("### 🧠 Cognitive Load & Technostress Micro-Survey")
    st.caption("3 Targeted Items • Privacy-Preserving • Instant Real-Time Manifold Positioning")
    
    if url_group:
        st.markdown(f'<div class="group-badge">🔒 Active Cohort: <b>{url_group}</b></div>', unsafe_allow_html=True)
        assigned_group = url_group
    else:
        assigned_group = st.text_input("Cohort Identifier (leave empty for general sample):", value="General").strip()
        if not assigned_group:
            assigned_group = "General"

    q1_opts = {
        "1. Fully energized, calm, and well-rested": 1.0,
        "2. Mild fatigue but rapid physical recovery": 2.0,
        "3. Occasional exhaustion following prolonged work": 3.0,
        "4. Chronic fatigue, emotional depletion, and low motivation": 4.0,
        "5. Severe persistent exhaustion and cognitive depletion": 5.0
    }
    q1_sel = st.radio("1. Post-work affective exhaustion and energy depletion (F1 - Burnout):", 
                      list(q1_opts.keys()), index=1, key="rad_q1", on_change=on_select_q1)
    val_f1 = q1_opts[q1_sel]

    q2_opts = {
        "1. Effortless mastery, zero stress from digital tool updates": 1.0,
        "2. Occasional mild adaptation friction with new software": 2.0,
        "3. High-paced digital acceleration and software turnover": 3.0,
        "4. Frequent stress and overload from emerging software and AI tools": 4.0,
        "5. Overwhelmed, chronic cognitive friction and continuous pressure": 5.0
    }
    q2_sel = st.radio("2. Stress and adaptation friction from new software and AI tools (F2 - AI Technostress):", 
                      list(q2_opts.keys()), index=2, key="rad_q2", on_change=on_select_q2)
    val_f2 = q2_opts[q2_sel]

    q3_opts = {
        "1. Proactive, agile execution with robust self-regulation": 1.0,
        "2. Stable cognitive adaptation with minimal coping delays": 2.0,
        "3. Occasional hesitation and procedural uncertainty": 3.0,
        "4. Frequent coping hesitation, task deferral, and procrastination": 4.0,
        "5. Severe adaptive paralysis and acute self-regulation deficits": 5.0
    }
    q3_sel = st.radio("3. Self-regulation and decision hesitation under digital demands (F3 - Latency):", 
                      list(q3_opts.keys()), index=1, key="rad_q3", on_change=on_select_q3)
    val_f3 = q3_opts[q3_sel]

    st.write("")
    btn_submit = st.button("🚀 Submit & Project Coordinates", use_container_width=True, type="primary")

    if btn_submit:
        t_end = time.time()
        dt_end = datetime.now()
        
        t0 = st.session_state.t0
        t1 = st.session_state.t_q1 if st.session_state.t_q1 else t0 + (t_end - t0) * 0.33
        t2 = st.session_state.t_q2 if st.session_state.t_q2 else t1 + (t_end - t1) * 0.5
        t3 = st.session_state.t_q3 if st.session_state.t_q3 else t2 + (t_end - t2) * 0.5

        dt_q1 = round(max(0.5, t1 - t0), 1)
        dt_q2 = round(max(0.5, t2 - t1), 1)
        dt_q3 = round(max(0.5, t3 - t2), 1)
        dt_total = round(t_end - t0, 1)

        # Chuẩn hóa thời gian suy xét logarit kết hợp câu hỏi điều hòa (Cognitive Latency Proxy)
        time_score = 1.0 + 4.0 * min(1.0, max(0.0, (np.log(1 + dt_total) - np.log(6)) / (np.log(45) - np.log(6))))
        final_f3 = round(0.5 * (val_f3 + time_score), 2)

        record = {
            'timestamp': dt_end.strftime("%Y-%m-%d %H:%M:%S"),
            'date': dt_end.strftime("%Y-%m-%d"),
            'year': dt_end.year,
            'month': dt_end.month,
            'day': dt_end.day,
            'start_time': st.session_state.dt_start.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': dt_end.strftime("%Y-%m-%d %H:%M:%S"),
            'group_id': assigned_group,
            'Burnout_Level': val_f1,
            'AI_Technostress': val_f2,
            'Cognitive_Latency_Proxy': final_f3,
            'Coping_Score': val_f3,
            't1_sec': dt_q1,
            't2_sec': dt_q2,
            't3_sec': dt_q3,
            'total_latency_sec': dt_total
        }

        with st.spinner("Recording responses and projecting manifold coordinates..."):
            commit_to_github(record)

        st.session_state.latest_user = record
        st.query_params["view"] = "result"
        if assigned_group:
            st.query_params["group"] = assigned_group
            
        st.session_state.dt_start = datetime.now()
        st.session_state.t0 = time.time()
        st.session_state.t_q1 = None
        st.session_state.t_q2 = None
        st.session_state.t_q3 = None
        st.rerun()

# ==============================================================================
# VIEW 2: MANIFOLD POSITIONING & 3-COLUMN COMPACT DASHBOARD
# ==============================================================================
else:
    c_head1, c_head2 = st.columns([1.5, 4.5])
    with c_head1:
        if st.button("⬅️ Retake Survey", use_container_width=True):
            st.query_params["view"] = "survey"
            st.session_state.dt_start = datetime.now()
            st.session_state.t0 = time.time()
            st.rerun()

    df_resp, _ = get_github_data()

    if not df_resp.empty:
        if st.session_state.latest_user is None:
            st.session_state.latest_user = df_resp.iloc[-1].to_dict()

        u_f1 = float(st.session_state.latest_user.get('Burnout_Level', 2.0))
        u_f2 = float(st.session_state.latest_user.get('AI_Technostress', 2.0))
        u_f3 = float(st.session_state.latest_user.get('Cognitive_Latency_Proxy', 2.0))
        u_grp = str(st.session_state.latest_user.get('group_id', url_group if url_group else "General"))
        u_t1 = float(st.session_state.latest_user.get('t1_sec', 2.5))
        u_t2 = float(st.session_state.latest_user.get('t2_sec', 3.0))
        u_t3 = float(st.session_state.latest_user.get('t3_sec', 2.0))
        u_total = float(st.session_state.latest_user.get('total_latency_sec', 7.5))

        available_groups = ["All Cohorts"] + sorted(list(df_resp['group_id'].dropna().astype(str).unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        with c_head2:
            chosen_grp = st.selectbox("Filter Cohort Analytics:", available_groups, index=sel_idx)

        if chosen_grp == "All Cohorts":
            df_plot = df_resp
            grp_name = "All Participants"
        else:
            df_plot = df_resp[df_resp['group_id'].astype(str) == chosen_grp]
            grp_name = chosen_grp

        # --- OUT-OF-SAMPLE TOPOLOGICAL PROJECTION ---
        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        u_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
        u_2d = pca.transform(u_scaled)[0]

        fig = go.Figure()
        # Layer 1: Global Normative Baseline
        fig.add_trace(go.Scatter(
            x=df_bg_sample['PC1'], y=df_bg_sample['PC2'],
            mode='markers', marker=dict(size=4.5, color='#cbd5e1', opacity=0.4),
            name='Global Norms (N=3,459)', hoverinfo='skip'
        ))
        # Layer 2: Cohort Members
        fig.add_trace(go.Scatter(
            x=sample_2d[:, 0], y=sample_2d[:, 1],
            mode='markers', marker=dict(size=7, color='#f43f5e', opacity=0.8),
            name=f'Cohort: {grp_name}', hoverinfo='name'
        ))
        # Layer 3: Cohort Centroid (Star)
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[c_grp[0]], y=[c_grp[1]],
                mode='markers', marker=dict(symbol='star', size=15, color='#fbbf24', line=dict(color='black', width=1.2)),
                name=f'Centroid: {grp_name}', hoverinfo='name'
            ))
        # Layer 4: Individual Position
        fig.add_trace(go.Scatter(
            x=[u_2d[0]], y=[u_2d[1]],
            mode='markers+text', marker=dict(size=16, color='#06b6d4', line=dict(color='#083344', width=2.2)),
            text=["📍 Your State"], textposition="top center",
            textfont=dict(color="#083344", size=11), name='Your Coordinates', hoverinfo='text'
        ))

        fig.update_layout(
            margin=dict(l=5, r=5, t=10, b=20),
            height=280,
            dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5, font=dict(size=8.5)),
            template="plotly_white",
            xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", showgrid=True, zeroline=False),
            yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", showgrid=True, zeroline=False)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})
        st.caption("🔍 *Tip:* Pinch or scroll to zoom • Double-tap to reset view.")

        # --- DIAGNOSTIC CLASSIFICATION (CLEAN MARKDOWN WITHOUT RAW ASTERISKS) ---
        if u_f2 >= 3.5 and u_f1 >= 3.5:
            p_title = "High Digital Acceleration & Exhaustion"
            p_desc = "Intensive professional workloads combined with rapid AI tool turnover impede restorative recovery."
            p_color = "#e11d48"
            tips_html = "<li>Enforce strict asynchronous digital boundaries.</li><li>Integrate 5-minute micro-breaks every 45 minutes of screen time.</li><li>Prune low-priority tasks and focus on linear execution.</li>"
        elif u_f2 >= 3.5 and u_f1 < 3.5:
            p_title = "Elevated Technological Adaptation Stress"
            p_desc = "Good physiological energy reserves, but continuous digital learning curves induce acute cognitive friction."
            p_color = "#d97706"
            tips_html = "<li>Standardize on 1–2 indispensable core AI tools.</li><li>Leverage peer tips and cheat-sheets to bypass trial-and-error.</li><li>Allow yourself a phased, natural adaptation window.</li>"
        elif u_f1 >= 3.5 and u_f2 < 3.5:
            p_title = "Occupational Fatigue Requiring Rest"
            p_desc = "Overall administrative or physical volume is demanding, though not primarily driven by AI churn."
            p_color = "#ea580c"
            tips_html = "<li>Prioritize restorative sleep hygiene.</li><li>Negotiate extensions on non-critical administrative deadlines.</li><li>Incorporate light physical activity and outdoor walks.</li>"
        else:
            p_title = "Sustainable Adaptive Equilibrium"
            p_desc = "You maintain a resilient balance between daily workload and software adaptation demands."
            p_color = "#16a34a"
            tips_html = "<li>Maintain your current healthy work-rest boundaries.</li><li>Share workflow strategies with departmental peers.</li><li>Remain mindful during peak exam or grading cycles.</li>"

        # Đánh giá độ trễ nhận thức từng câu hỏi
        max_t = max(u_t1, u_t2, u_t3)
        if max_t == u_t2 and u_t2 > 5.0:
            time_remark = "Longest deliberation on Item 2 (AI Technostress), indicating high cognitive appraisal on digital tools."
        elif max_t == u_t1 and u_t1 > 5.0:
            time_remark = "Longest pause on Item 1 (Fatigue), reflecting significant awareness of bodily exhaustion."
        elif max_t == u_t3 and u_t3 > 5.0:
            time_remark = "Longest delay on Item 3 (Self-regulation), indicating hesitation in active coping mechanisms."
        else:
            time_remark = "Decisive, consistent response velocity across all assessment items."

        avg_f1 = df_plot['Burnout_Level'].mean()
        avg_f2 = df_plot['AI_Technostress'].mean()
        grp_summary = f"Mean Burnout: <b>{avg_f1:.2f}/5.0</b> | Mean Technostress: <b>{avg_f2:.2f}/5.0</b>"

        # --- 3-COLUMN COMPACT LAYOUT (IN SINGLE ROW, ZERO OVERFLOW) ---
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid {p_color};">
                <b style="color: {p_color}; font-size: 0.95rem;">👤 1. Individual State</b><br>
                <b>Diagnostic:</b> {p_title}<br>
                <span style="color: #475569;">{p_desc}</span><br><br>
                <b>Responses & Decision Latencies:</b><br>
                • Item 1 (Burnout): <b>{u_f1:.1f} pts</b> ({u_t1}s)<br>
                • Item 2 (AI Stress): <b>{u_f2:.1f} pts</b> ({u_t2}s)<br>
                • Item 3 (Latency): <b>{u_f3:.1f} pts</b> ({u_t3}s)<br>
                • Total Latency: <b>{u_total}s</b><br>
                <span style="color: #0369a1; font-size: 0.82rem;">{time_remark}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid #6366f1;">
                <b style="color: #4f46e5; font-size: 0.95rem;">👥 2. Cohort Benchmark</b><br>
                <b>Cohort:</b> {grp_name} ({len(df_plot)} responses)<br>
                {grp_summary}<br><br>
                <b>Collective Manifold Trend:</b><br>
                The cohort centroid is positioned in the <b>{'High Load' if avg_f2 >= 3.5 else 'Equilibrium'}</b> regime relative to the global baseline.
            </div>
            """, unsafe_allow_html=True)

        with col_c3:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid #10b981;">
                <b style="color: #059669; font-size: 0.95rem;">💡 3. Coping Recommendations</b><br>
                <b>For You:</b>
                <ul style="margin: 2px 0 6px 0; padding-left: 16px; color: #334155;">
                    {tips_html}
                </ul>
                <b>For Department Chairs:</b><br>
                <span style="color: #334155;">Promote phased AI rollouts and shared templates to prevent individual trial-and-error exhaustion.</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No cohort data available yet. Please return and submit the initial assessment.")
