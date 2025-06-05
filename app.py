import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

# 设置页面标题
st.set_page_config(page_title="抑郁症预测模型", layout="wide")
st.title("基于睡眠特征的抑郁症风险预测")


# 加载模型和标准化器
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('standard_scaler.pkl')
        feature_ranges = joblib.load('feature_ranges.pkl')
        background = joblib.load('shap_background.pkl')
        explainer = shap.TreeExplainer(model, background)

        return {
            'model': model,
            'scaler': scaler,
            'feature_ranges': feature_ranges,
            'explainer': explainer
        }
    except Exception as e:
        st.error(f"加载资源失败: {str(e)}")
        return None


resources = load_resources()

if not resources:
    st.stop()

# 侧边栏 - 用户输入
st.sidebar.header("输入睡眠特征数据")

# 特征描述
feature_descriptions = {
    'psqi': "PSQI睡眠质量指数（分数越高表示睡眠质量越差）",
    'flinder': "Flinders疲劳量表评分（分数越高表示疲劳程度越严重）",
    'remallused_theta_a': "REM期Theta脑电波相对功率（与认知功能相关）"
}

# 创建输入控件
user_inputs = {}
for feature, info in resources['feature_ranges'].items():
    description = feature_descriptions.get(feature, feature)
    if feature == 'remallused_theta_a':
        # 小数型slider
        user_inputs[feature] = st.sidebar.slider(
            description,
            min_value=float(info['min']),
            max_value=float(info['max']),
            value=float(info['mean']),
            step=0.01,      # 你可以改成你想要的小数步长
            format="%.3f"
        )
    else:
        # 整数型slider
        user_inputs[feature] = st.sidebar.slider(
            description,
            min_value=int(info['min']),
            max_value=int(info['max']),
            value=int(round(info['mean'])),
            step=1           # 你可以改成你想要的整数步长
        )
# 创建预测按钮
predict_button = st.sidebar.button("预测抑郁风险", type="primary")

# 主内容区域
if predict_button:
    # 创建输入数据框
    input_data = pd.DataFrame([user_inputs])

    # 标准化输入数据
    input_scaled = resources['scaler'].transform(input_data)

    # 预测概率
    prob_depression = resources['model'].predict_proba(input_scaled)[0][1]
    risk_percentage = prob_depression * 100

    # 显示结果
    st.subheader("预测结果")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="抑郁风险概率", value=f"{prob_depression:.2%}")

        # 进度条可视化
        st.progress(int(risk_percentage))
        st.caption(f"风险程度: {risk_percentage:.1f}%")

    with col2:
        # 风险水平解释
        if prob_depression < 0.3:
            st.success("**低风险**: 您的睡眠特征显示较低的抑郁风险")
            st.info("建议: 继续保持良好的睡眠习惯")
        elif prob_depression < 0.7:
            st.warning("**中等风险**: 您的睡眠特征显示中等抑郁风险")
            st.info("建议: 关注睡眠质量，考虑咨询专业人士")
        else:
            st.error("**高风险**: 您的睡眠特征显示较高的抑郁风险")
            st.info("建议: 请及时咨询心理健康专业人士")

    # 分隔线
    st.divider()

    # 解释性分析
    st.subheader("特征影响分析")

    # 计算SHAP值
    shap_values = resources['explainer'].shap_values(input_scaled)

    # 处理SHAP值维度
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_sample = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_sample = shap_values[0, :, 1]
    else:
        shap_values_sample = shap_values[0]

    # 创建特征影响图表
    fig, ax = plt.subplots(figsize=(10, 4))
    features = list(feature_descriptions.keys())
    feature_names = [feature_descriptions[f].split("（")[0] for f in features]

    # 创建条形图
    colors = ['#FF6B6B' if val > 0 else '#4ECDC4' for val in shap_values_sample]
    bars = ax.barh(feature_names, shap_values_sample, color=colors)
    ax.set_xlabel('对抑郁风险的影响值')
    ax.set_title('各睡眠特征对预测结果的影响')
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        label_x = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                va='center', ha='left' if width > 0 else 'right',
                fontsize=10)

    st.pyplot(fig)

    # 添加解释说明
    st.markdown("""
    **影响值说明:**
    - **正值（红色）**: 增加抑郁风险
    - **负值（绿色）**: 降低抑郁风险
    - 绝对值越大表示影响越显著
    """)

    # 添加具体特征解释
    st.divider()
    st.subheader("特征详细说明")

    for feature in features:
        with st.expander(f"{feature_descriptions[feature].split('（')[0]}"):
            st.markdown(f"**当前值:** {user_inputs[feature]:.3f}")
            st.markdown(
                f"**正常范围:** {resources['feature_ranges'][feature]['min']:.2f} - {resources['feature_ranges'][feature]['max']:.2f}")
            st.markdown(f"**解释:** {feature_descriptions[feature].split('（')[1][:-1]}")

# 添加应用说明（仅在初始页面显示）
if not predict_button:
    st.markdown("""
    ## 欢迎使用抑郁症风险预测工具

    本工具基于机器学习模型，通过分析您的睡眠特征来评估抑郁风险。研究表明，睡眠质量与心理健康密切相关。

    **使用方法:**
    1. 在左侧边栏调整各项睡眠特征值
    2. 点击"预测抑郁风险"按钮
    3. 查看预测结果和详细分析

    **注意事项:**
    - 本工具仅提供参考信息，不能替代专业医疗诊断
    - 预测结果基于统计模型，可能存在误差
    - 如有心理健康问题，请咨询专业医生

    *数据来源: 临床睡眠研究数据集*
    """)

    # 添加模型信息
    st.sidebar.markdown("""
    **模型信息:**
    - 算法: 随机森林
    - 特征数: 3
    - 验证AUC: 0.86
    """)

# 添加页脚
st.sidebar.divider()
st.sidebar.caption("© 2025 汕头大学睡眠医学中心 | 测试版本 1.0")
