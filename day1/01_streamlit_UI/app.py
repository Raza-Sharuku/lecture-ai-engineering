import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import os

# ページ設定
st.set_page_config(
    page_title="AI ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.title("AI ダッシュボード")
    st.markdown("---")
    
    # ナビゲーションメニュー
    page = st.radio(
        "メニュー",
        ["📊 ダッシュボード", "⚙️ 設定"]
    )

# メインコンテンツ
if page == "📊 ダッシュボード":
    # ヘッダー
    st.title("AI パフォーマンスダッシュボード")
    
    # メトリクスカード
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("精度", "95.2%", "+2.1%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.metric("処理時間", "0.8秒", "-0.2秒")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.metric("リクエスト数", "1,234", "+123")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.metric("エラー率", "0.5%", "-0.1%")
        st.markdown('</div>', unsafe_allow_html=True)

    # グラフ
    st.markdown("### パフォーマンストレンド")
    col1, col2 = st.columns(2)
    
    with col1:
        # ラインチャート
        x = list(range(20))  # 0から19までの数値
        chart_data = pd.DataFrame({
            '時間': x,
            '精度': [0.95 + 0.02 * np.random.randn() for _ in range(20)],
            'リコール': [0.92 + 0.03 * np.random.randn() for _ in range(20)],
            'F1スコア': [0.93 + 0.02 * np.random.randn() for _ in range(20)]
        })
        st.line_chart(chart_data.set_index('時間'))
    
    with col2:
        # バーチャート
        data = pd.DataFrame({
            'カテゴリ': ['A', 'B', 'C', 'D'],
            '値': [30, 45, 25, 60]
        })
        st.bar_chart(data.set_index('カテゴリ'))

    # インタラクティブな機能
    st.markdown("### モデル設定")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "モデルタイプ",
            ["Random Forest", "gemma", "LightGBM", "Neural Network"]
        )
        learning_rate = st.slider("学習率", 0.01, 0.1, 0.05, 0.01)
    
    with col2:
        batch_size = st.number_input("バッチサイズ", 32, 512, 128, 32)
        epochs = st.slider("エポック数", 10, 100, 50, 10)

    if st.button("モデルをトレーニング"):
        with st.spinner("トレーニング中..."):
            time.sleep(2)
            st.success("トレーニング完了！")

else:  # 設定ページ
    st.title("設定")
    
    # アプリケーション設定
    st.subheader("アプリケーション設定")
    theme = st.selectbox("テーマ", ["Light", "Dark"])
    language = st.selectbox("言語", ["日本語", "English"])
    
    # 通知設定
    st.subheader("通知設定")
    email_notifications = st.checkbox("メール通知を有効にする")
    if email_notifications:
        email = st.text_input("メールアドレス")
    
    # 保存ボタン
    if st.button("設定を保存"):
        st.success("設定が保存されました！")

