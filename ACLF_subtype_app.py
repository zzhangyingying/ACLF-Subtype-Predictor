import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Authentic Trajectory Data from R (Days 1-7) ---
SUBTYPES = {
    'Subtype 1': {
        'tbil': [1.8065, 2.348, 2.8896, 3.4312, 3.9727, 4.5143, 5.0558],
        'inr':  [1.6049, 1.555, 1.5335, 1.5401, 1.5751, 1.6383, 1.7297],
        'color': '#2ecc71', 'risk': 'Low Risk'
    },
    'Subtype 2': {
        'tbil': [9.5905, 9.9284, 10.2662, 10.604, 10.9418, 11.2796, 11.6175],
        'inr':  [2.8281, 2.7252, 2.6392, 2.5699, 2.5173, 2.4815, 2.4625],
        'color': '#f1c40f', 'risk': 'Intermediate-High Risk'
    },
    'Subtype 3': {
        'tbil': [26.2633, 26.2046, 25.9897, 25.6186, 25.0913, 24.4078, 23.568],
        'inr':  [2.7383, 2.6395, 2.559, 2.4966, 2.4524, 2.4264, 2.4186],
        'color': '#e74c3c', 'risk': 'Very High Risk'
    },
    'Subtype 4': {
        'tbil': [3.8982, 4.0421, 4.2874, 4.6342, 5.0825, 5.6322, 6.2834],
        'inr':  [2.2284, 2.1148, 2.036, 1.9921, 1.9832, 2.0091, 2.07],
        'color': '#3498db', 'risk': 'Intermediate-Low Risk'
    }
}

st.set_page_config(page_title="ACLF Subtype Predictor", layout="wide")

# --- 2. Sidebar: 7-Day Flexible Input ---
with st.sidebar:
    st.header("📋 7-Day Clinical Registry")
    st.markdown("Enter data for available days (Min. 3 days required):")
    
    # 用一个字典来存储录入的值，初始化为 None
    inputs = {'day': [], 'tbil': [], 'inr': []}
    
    for i in range(1, 8):
        with st.expander(f"Day {i}", expanded=(i<=3)):
            col1, col2 = st.columns(2)
            # 使用 0.0 作为初始值，但在后面判断是否被修改
            t_val = col1.number_input(f"TBil", value=0.0, step=0.1, key=f"t{i}", format="%.1f")
            i_val = col2.number_input(f"INR", value=0.0, step=0.1, key=f"i{i}", format="%.2f")
            
            # 只有当 TBil 和 INR 都不为 0 时，才视作有效输入
            if t_val > 0 and i_val > 0:
                inputs['day'].append(i)
                inputs['tbil'].append(t_val)
                inputs['inr'].append(i_val)

    st.markdown("---")
    # 逻辑判断：有效天数是否大于等于3
    valid_days_count = len(inputs['day'])
    if valid_days_count < 3:
        st.warning(f"⚠️ Need data for {3 - valid_days_count} more day(s)")
        predict_btn = st.button("🚀 Match Subtype", disabled=True)
    else:
        st.success(f"✅ {valid_days_count} days of data ready")
        predict_btn = st.button("🚀 Match Subtype", disabled=False)

# --- 3. Main Interface ---
st.title("🏥 ACLF Subtype Trajectory Matcher")
st.caption("Advanced LCMA-based Clinical Phenotyping Tool")

if predict_btn:
    # 算法：只对用户填写的特定天数进行平方误差计算
    scores = {}
    user_days = np.array(inputs['day']) - 1 # 转换为索引 0-6
    u_t = np.array(inputs['tbil'])
    u_i = np.array(inputs['inr'])

    for name, data in SUBTYPES.items():
        # 提取标准曲线对应天数的数值
        ref_t = np.array(data['tbil'])[user_days]
        ref_i = np.array(data['inr'])[user_days]
        
        # 计算欧氏距离
        dist = np.sqrt(np.sum((u_t - ref_t)**2) + np.sum((u_i - ref_i)**2))
        scores[name] = dist
    
    best_sub = min(scores, key=scores.get)
    res = SUBTYPES[best_sub]
    
    # 展示结果
    st.markdown(f"""
        <div style="padding:20px; border-radius:12px; background-color:white; border-left:10px solid {res['color']}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="color:{res['color']}; margin:0;">Classification: {best_sub}</h2>
            <p style="font-size:1.2em;">Prognostic Layer: <b>{res['risk']}</b></p>
            <p>Trajectory matched based on data from Days: {', '.join(map(str, inputs['day']))}.</p>
        </div>
    """, unsafe_allow_html=True)

    # 绘图
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=("TBil Trend", "INR Trend"))
    days_full = [1, 2, 3, 4, 5, 6, 7]
    
    for name, data in SUBTYPES.items():
        is_match = (name == best_sub)
        op = 0.95 if is_match else 0.1
        fig.add_trace(go.Scatter(x=days_full, y=data['tbil'], name=name, line=dict(color=data['color'], width=4 if is_match else 1), opacity=op), row=1, col=1)
        fig.add_trace(go.Scatter(x=days_full, y=data['inr'], name=name, line=dict(color=data['color'], width=4 if is_match else 1), opacity=op, showlegend=False), row=1, col=2)

    # 叠加用户离散数据
    fig.add_trace(go.Scatter(x=inputs['day'], y=inputs['tbil'], name="Patient", mode='markers+lines', marker=dict(size=12, color='black', symbol='x'), line=dict(dash='dot', color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=inputs['day'], y=inputs['inr'], name="Patient", mode='markers+lines', marker=dict(size=12, color='black', symbol='x'), line=dict(dash='dot', color='black'), showlegend=False), row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("💡 **Clinical Tip:** You can enter any 3 or more days of data within the first week (e.g., Day 1, 4, 7). The algorithm will align the available points to the reference subphenotype trajectories.")