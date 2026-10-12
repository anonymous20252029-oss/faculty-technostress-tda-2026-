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

# Nhúng CSS tối ưu giao diện trên thiết bị di động
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3rem; }
</style>
""", unsafe_allow_html=True)

# Lấy mã nhóm từ query parameters trên URL (Ví dụ: ?group=CNTT_TDTU)
query_params = st.query_params
group_id = query_params.get("group", "CNTT_TDTU")

st.title("🧠 Đánh Giá Ma Sát Nhận Thức & Áp Lực Công Nghệ")
st.markdown(f"**Khảo sát vi mô tại chỗ | Nhóm khảo sát:** `{group_id}`")

# Khởi tạo bộ đo thời gian ẩn (Silent Latency Timers)
if "init_time" not in st.session_state:
    st.session_state.init_time = time.time()
if "t1_start" not in st.session_state:
    st.session_state.t1_start = time.time()

# Nạp nền chuẩn toàn cầu (N=3,459)
df_base, X_base, pca, X_pca_base = load_or_generate_baseline()

col_survey, col_vis = st.columns([1, 1.2])

with col_survey:
    st.subheader("1. Khảo Sát Vi Mô (3 Câu Hỏi)")
    with st.form("survey_form"):
        q1 = st.slider(
            "1. Mức độ kiệt quệ & hao mòn năng lượng sau giờ làm việc (F1 - Burnout):",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5
        )
        q2 = st.slider(
            "2. Áp lực & choáng ngợp trước nhịp độ đổi mới công cụ AI (F2 - Technostress):",
            min_value=1.0, max_value=5.0, value=3.5, step=0.5
        )
        q3 = st.slider(
            "3. Mức độ đắn đo, do dự khi phải xử lý đa nhiệm số (F3 - Latency Proxy):",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5
        )
        
        submitted = st.form_submit_button("🚀 Gửi Đánh Giá & Định Vị Tức Thời")

    if submitted:
        total_latency = round(time.time() - st.session_state.init_time, 2)
        st.session_state.user_vector = [q1, q2, q3]
        st.session_state.total_latency = total_latency

        # Ghi nhận kết quả
        record = {
            "Burnout_Level": q1,
            "AI_Technostress": q2,
            "Cognitive_Latency_Proxy": q3,
            "group_id": group_id,
            "total_latency_sec": total_latency,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        save_pilot_response(record)
        st.success(f"✅ Đã định vị thành công! Tổng thời gian suy xét: {total_latency} giây.")

with col_vis:
    st.subheader("2. Chiếu Tọa Độ Trực Tiếp Lên Đa Tạp Toàn Cầu")

    fig = go.Figure()

    # Lớp nền chuẩn toàn cầu N=3,459
    fig.add_trace(go.Scattergl(
        x=X_pca_base[:, 0],
        y=X_pca_base[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color=X_base[:, 1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Áp lực AI (F2)", len=0.7)
        ),
        name="Chuẩn Toàn Cầu (N=3,459)",
        opacity=0.35
    ))

    # Điểm tọa độ của người tham gia vừa gửi
    if "user_vector" in st.session_state:
        u_coords = project_out_of_sample(st.session_state.user_vector, pca)
        fig.add_trace(go.Scatter(
            x=[u_coords[0]],
            y=[u_coords[1]],
            mode='markers+text',
            marker=dict(size=18, color='#e11d48', symbol='star', line=dict(color='white', width=2)),
            text=["Bạn ở đây"],
            textposition="top center",
            name="Vị trí của bạn"
        ))

    fig.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

# Khuyến nghị hành động 3 lớp
if "user_vector" in st.session_state:
    u = st.session_state.user_vector
    st.divider()
    st.subheader("💡 Phản Hồi Cá Nhân Hóa & Khuyến Nghị Hành Động")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Kiệt Quệ (F1)", f"{u[0]:.1f} / 5.0", delta=f"{u[0] - 3.09:+.2f} vs Chuẩn")
    with col_b:
        st.metric("Áp Lực AI (F2)", f"{u[1]:.1f} / 5.0", delta=f"{u[1] - 3.38:+.2f} vs Chuẩn")
    with col_c:
        st.metric("Trễ Nhận Thức (F3)", f"{u[2]:.1f} / 5.0", delta=f"{u[2] - 2.86:+.2f} vs Chuẩn")

    if u[1] >= 4.0:
        st.warning("⚠️ **Cảnh báo Áp Lực Công Nghệ:** Tọa độ của bạn nằm ở bán cầu trên (tiệm cận nhóm kỹ sư doanh nghiệp IT). Khuyến nghị thiết lập ranh giới số, áp dụng vi nghỉ ngơi (micro-breaks) và giãn cách thời gian cập nhật phần mềm mới.")
    else:
        st.info("🌱 **Trạng Thái Thích Ứng Tốt:** Mức độ thích ứng công nghệ của bạn đang duy trì trong dải cân bằng an toàn.")
