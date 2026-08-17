import pandas as pd
import numpy as np
import PyPDF2
import google.generativeai as genai

JOB_DATA = {
    "Pozisyon": ["Data Scientist", "Backend Developer", "Frontend Developer", "AI Engineer"],
    "Gerekli_Yetenekler": [
        "python pandas scikit-learn machine learning sql data analysis deep learning",
        "python django flask postgresql docker rest api microservices git",
        "javascript react html css typescript tailwind redux rest api",
        "python pytorch tensorflow llm rag transformers nlp docker api"
    ]
}

def get_job_df():
    return pd.DataFrame(JOB_DATA)

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_match_score(cv_text, job_skills):
    if not cv_text or not job_skills:
        return 0.0
        
    cv_text_clean = cv_text.lower().replace('/', ' ').replace('-', ' ').replace(',', ' ')
    skills_list = [skill.strip().lower() for skill in job_skills.split() if skill.strip()]
    
    if not skills_list:
        return 0.0
        
    matched_count = 0
    for skill in skills_list:
        if skill in cv_text_clean:
            matched_count += 1
            
    score = (matched_count / len(skills_list)) * 100
    return round(score, 2)

def generate_llm_insights(api_key, cv_text, job_title, job_skills):
    """2. Sekme: Yetenek Açığı ve Gelişim Rotası"""
    if not api_key:
        return "⚠️ Lütfen sol menüden geçerli bir Gemini API Key girin."
        
    try:
        genai.configure(api_key=api_key.strip())
        # En kararlı ve güncel model ismi
        model = genai.GenerativeModel('gemini-3.5-flash')
        
        prompt = f"""
        Aday CV Metni:
        {cv_text}
        
        Hedef Pozisyon: {job_title}
        Pozisyon Gereksinimleri: {job_skills}
        
        Lütfen adayın kariyer gelişimi için Türkçe bir Rapor hazırla:
        1. **Eksik Yetenekler ve Gap Analizi**: CV'de eksik olan teknolojiler.
        2. **Kişiselleştirilmiş Öğrenme Rotası**: 1-3 aylık somut öğrenme adımları ve proje tavsiyeleri.
        3. **Güçlü Yönler**: Pozisyona doğrudan katkı sağlayacak yetkinlikler.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Hatanın gerçek nedenini ekrana basarak teşhisi kolaylaştırıyoruz
        return f"❌ Gemini API Hatası: {str(e)}"

def generate_interview_questions(api_key, cv_text, job_title, job_skills):
    """3. Sekme: İK ve Teknik Mülakat Soruları"""
    if not api_key:
        return "⚠️ Lütfen sol menüden geçerli bir Gemini API Key girin."
        
    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-3.5-flash')
        
        prompt = f"""
        Aday CV Metni:
        {cv_text}
        
        Hedef Pozisyon: {job_title}
        Pozisyon Gereksinimleri: {job_skills}
        
        Bir İK Uzmanı ve Kıdemli Teknik Lider gibi davranarak Türkçe Mülakat Seti oluştur:
        1. **Teknik Sorular (3 Adet)**: CV'de iddia ettiği araçlar ve pozisyon gereksinimleriyle ilgili ölçücü teknik sorular.
        2. **Davranışsal / Yetkinlik Soruları (2 Adet)**: Geçmiş tecrübelerini değerlendiren durum soruları.
        3. **Mülakatçı İpuçları**: İK'nın yanıtlarda beklemesi gereken ideal anahtar kelimeler/cevaplar.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Hatası: {str(e)}"
