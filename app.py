import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 한글 폰트 설정 (윈도우 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="FDA 리콜 GMP 리스크 분석", layout="wide")
st.title("FDA 의약품 리콜 GMP 리스크 분석 대시보드")

df = pd.read_csv('dashboard_data.csv')

with open('model_text.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

tab1, tab2 = st.tabs(["트렌드 대시보드", "위험도 예측"])

with tab1:
    st.subheader("GMP 카테고리별 리콜 분포")
    category_counts = df['gmp_category'].value_counts()
    fig, ax = plt.subplots()
    category_counts.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    st.subheader("카테고리별 Class I(고위험) 비율")
    risk_table = pd.crosstab(df['gmp_category'], df['classification'], normalize='index') * 100
    st.dataframe(risk_table.round(1))

with tab2:
    st.subheader("리콜 사유 입력하면 위험도 예측")
    user_input = st.text_area("리콜 사유를 영어로 입력하세요",
                                "Microbial contamination found during routine testing")

    if st.button("위험도 계산"):
        X_new = tfidf.transform([user_input])
        risk_prob = model.predict_proba(X_new)[0][1]
        st.metric("Class I(고위험) 확률", f"{risk_prob*100:.1f}%")

        if risk_prob >= 0.5:
            st.error("고위험군으로 예측됨")
        else:
            st.success("상대적 저위험군으로 예측됨")