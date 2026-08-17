import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app_backend import get_job_df, extract_text_from_pdf, calculate_match_score, generate_llm_insights, generate_interview_questions

st.set_page_config(page_title="SkillSync AI", layout="wide")

st.title("🎯 SkillSync AI - Yetenek Açığı & İK Danışmanı")
st.caption("CV Analizi, ML Tabanlı Uyum Skoru ve LLM Destekli Mülakat Asistanı")

# Yan Menü (Sidebar) - Ayarlar ve Seçim
st.sidebar.header("⚙️ Ayarlar")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

df_jobs = get_job_df()
selected_job = st.sidebar.selectbox("Hedef Pozisyonu Seçin", df_jobs["Pozisyon"])
job_skills = df_jobs[df_jobs["Pozisyon"] == selected_job]["Gerekli_Yetenekler"].values[0]

uploaded_file = st.sidebar.file_uploader("CV Yükle (PDF)", type=["pdf"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 1. Aday Uyum Paneli", "🎯 2. Yetenek Açığı & Gelişim Rotası", "💬 3. İK Mülakat Asistanı", "📈 4. Model & Sistem Metrikleri"])

cv_text = ""
match_score = 0.0

if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    match_score = calculate_match_score(cv_text, job_skills)

# 1. Aday Uyum Paneli
with tab1:
    st.header("CV & Pozisyon Eşleşme Analizi")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(label="Pozisyon Uyum Skoru", value=f"%{match_score}")
        
        # SABİT RENK HARİTALAMA (Uyumlu -> Yeşil, Eksik -> Kırmızı)
        uyum_data = pd.DataFrame({
            "Durum": ["Uyumlu", "Eksik"],
            "Oran": [match_score, max(0.0, 100.0 - match_score)]
        })
        
        fig = px.pie(
            uyum_data, 
            names="Durum", 
            values="Oran", 
            hole=0.6,
            color="Durum",
            color_discrete_map={
                "Uyumlu": "#2ecc71",  # Sabit Yeşil
                "Eksik": "#e74c3c"    # Sabit Kırmızı
            }
        )
        fig.update_traces(sort=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write(f"**Seçilen Pozisyon:** {selected_job}")
        st.write(f"**Aranan Yetenekler:** {job_skills}")
        st.text_area("Okunan CV Metni Özeti", cv_text[:500] if cv_text else "Lütfen sol menüden bir PDF CV yükleyin.", height=200)

# 2. Yetenek Açığı & Gelişim Rotası
with tab2:
    st.subheader("🤖 AI Destekli Yetenek Açığı (Skill-Gap) Analizi")
    if uploaded_file and api_key:
        if st.button("Gelişim Rotası Oluştur"):
            with st.spinner("Gemini aday için gelişim haritası oluşturuyor..."):
                analysis = generate_llm_insights(api_key, cv_text, selected_job, job_skills)
                st.markdown(analysis)
    else:
        st.info("Lütfen sol menüden hem geçerli bir Gemini API Key girin hem de CV PDF'inizi yükleyin.")

# 3. İK Mülakat Asistanı
with tab3:
    st.subheader("💬 İK Mülakat Asistanı ve Soru Üretici")
    st.write("Adayın CV'si ve pozisyon gereksinimlerine özel üretilen mülakat rehberi.")
    
    if uploaded_file and api_key:
        if st.button("Adaya Özel Mülakat Soruları Üret"):
            with st.spinner("Gemini İK için teknik ve davranışsal mülakat soruları hazırlıyor..."):
                questions = generate_interview_questions(api_key, cv_text, selected_job, job_skills)
                st.markdown("### 📋 Mülakat Soruları & Değerlendirme İpuçları")
                st.markdown(questions)
    else:
        st.warning("Mülakat sorularını üretmek için lütfen sol menüden hem Gemini API Key girin hem de bir CV yükleyin.")

# 4. Model & Sistem Metrikleri (TAMAMEN DİNAMİK CANLI METRİKLER)
with tab4:
    st.subheader("📈 Model & Canlı Sistem Metrikleri")
    
    # Canlı hesaplanan dinamik değerler
    word_count = len(cv_text.split()) if cv_text else 0
    char_count = len(cv_text) if cv_text else 0
    api_status = "🟢 Bağlı & Aktif" if api_key else "🔴 Key Bekleniyor"
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mevcut Uyum Skoru", f"%{match_score}")
    col_b.metric("Aktif LLM Modeli", "Gemini 3.5 Flash")
    col_c.metric("API Durumu", api_status)
    col_d.metric("Okunan Kelime Sayısı", f"{word_count} Kelime")
    
    st.markdown("---")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("**🔍 Metin Analitiği Detayları**")
        st.write(f"- **Toplam Karakter Sayısı:** {char_count}")
        st.write(f"- **Taranan Pozisyon:** {selected_job}")
        st.write(f"- **Eşleşme Algoritması:** Direct Match / LLM Hybrid")
    
    with col_y:
        st.markdown("**⚡ Sistem Durumu**")
        st.write("- **Backend Engine:** Python / Pandas / PyPDF2")
        st.write("- **Frontend Framework:** Streamlit Wide Layout")
        st.write("- **AI Service Status:** Google Gemini API Integration")
