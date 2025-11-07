# 🎙️ IELTS Speaking Practice App

An AI-powered **IELTS Speaking Practice** web application built with **Python** and **Streamlit**.

This app helps users **practice speaking for IELTS** by:
- Recording or uploading speech 🎤
- Transcribing it to text 📝
- Analyzing **fluency, grammar, and vocabulary**
- Providing **feedback** and **approximate IELTS band scores**

---

## 🚀 Features

✅ Record live audio using your microphone  
✅ Automatic speech transcription (Google Speech Recognition / Sphinx fallback)  
✅ Intelligent text analysis with **spaCy NLP**  
✅ IELTS-style scoring for:
- Fluency & Coherence
- Grammar & Accuracy
- Vocabulary Range  
✅ Personalized feedback  
✅ Text-to-Speech playback of your spoken response  

---

## 🧠 Technologies Used

| Component | Library |
|------------|----------|
| Web UI | [Streamlit](https://streamlit.io) |
| Speech Recording | sounddevice |
| Speech Recognition | speech_recognition |
| Text-to-Speech | gTTS |
| Natural Language Processing | spaCy |
| Numerical Processing | NumPy, SciPy |

---

## 🖥️ Local Setup (Recommended for Full Functionality)

> 💡 Because this app uses a **microphone**, it must run locally — Streamlit Cloud does not support `sounddevice`.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/ielts-speaking-app.git
cd ielts-speaking-app
