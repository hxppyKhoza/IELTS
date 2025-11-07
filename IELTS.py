import os

import io

import sounddevice as sd

from scipy.io.wavfile import write

import speech_recognition as sr

from gtts import gTTS

import spacy

from collections import Counter

import tempfile

import streamlit as st

import numpy as np

import base64
 
# Load spaCy model

nlp = spacy.load("en_core_web_sm")
 
# -------------------------------

# AUDIO FUNCTIONS

# -------------------------------

def record_audio(filename="recorded_audio.wav", duration=10, sample_rate=16000):

    """Record audio using the microphone."""

    st.info("🎙️ Recording... Please speak clearly.")

    audio_data = sd.rec(int(duration * sample_rate),

                        samplerate=sample_rate, channels=1, dtype='int16')

    sd.wait()

    write(filename, sample_rate, audio_data)

    st.success(f"✅ Recording saved to {filename}")

    return filename
 
def transcribe_audio(file_path):

    """Transcribe recorded speech using SpeechRecognition."""

    r = sr.Recognizer()

    try:

        with sr.AudioFile(file_path) as source:

            r.adjust_for_ambient_noise(source)

            audio = r.record(source)

        try:

            transcription = r.recognize_google(audio)

            return transcription

        except sr.UnknownValueError:

            return "Could not understand the audio clearly. Please try again."

        except sr.RequestError:

            try:

                transcription = r.recognize_sphinx(audio)

                return transcription

            except:

                return "Transcription service unavailable. Check your internet connection."

    except Exception as e:

        return f"Error processing audio: {e}"
 
def text_to_speech(text):

    """Convert text to speech and return playable audio."""

    try:

        tts = gTTS(text=text, lang='en', slow=False)

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

        tts.save(temp_audio.name)

        return temp_audio.name

    except Exception as e:

        st.error(f"Text-to-speech failed: {e}")

        return None
 
# -------------------------------

# ANALYSIS FUNCTION

# -------------------------------

def analyze_transcription(transcription):

    """Analyze transcription for IELTS-like metrics."""

    if not transcription or transcription.startswith(("Could not", "Error")):

        return {

            "fluency": 0,

            "grammar_score": 0,

            "vocabulary_richness": 0,

            "total_score": 0,

            "feedback": "Unable to analyze - transcription failed"

        }
 
    doc = nlp(transcription)

    word_count = len([t for t in doc if t.is_alpha])

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    pos_counts = Counter([t.pos_ for t in doc if t.is_alpha])
 
    grammar_variety = len(pos_counts)

    verb_count = pos_counts.get("VERB", 0)

    noun_count = pos_counts.get("NOUN", 0)

    adj_count = pos_counts.get("ADJ", 0)
 
    unique_words = len(set([t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]))

    vocabulary_richness = unique_words / word_count if word_count > 0 else 0

    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
 
    # Scoring (0-9 IELTS scale approximation)

    fluency_score = min(9, (word_count / 10) + (avg_words_per_sentence / 5))

    grammar_score = min(9, grammar_variety * 0.5 + (verb_count + noun_count) / word_count * 9)

    vocab_score = min(9, vocabulary_richness * 9 + (adj_count / word_count) * 3)

    total_score = (fluency_score + grammar_score + vocab_score) / 3
 
    feedback_parts = []

    if fluency_score < 5:

        feedback_parts.append("🗣️ Try to speak more fluently and extend your responses.")

    if grammar_score < 5:

        feedback_parts.append("🧠 Use more varied grammar and sentence structures.")

    if vocab_score < 5:

        feedback_parts.append("📚 Include more diverse and descriptive vocabulary.")

    if not feedback_parts:

        feedback_parts.append("✅ Excellent! Keep practicing to maintain your level.")

    feedback = " ".join(feedback_parts)
 
    return {

        "fluency": fluency_score,

        "grammar_score": grammar_score,

        "vocabulary_richness": vocab_score,

        "total_score": total_score,

        "word_count": word_count,

        "sentence_count": sentence_count,

        "feedback": feedback

    }
 
# -------------------------------

# STREAMLIT APP

# -------------------------------

st.set_page_config(page_title="IELTS Speaking Practice", page_icon="🎤", layout="centered")
 
st.title("🎤 IELTS Speaking Practice App")

st.write("This tool helps you **practice speaking** for IELTS by analyzing your fluency, grammar, and vocabulary.")
 
duration = st.slider("Recording Duration (seconds):", 5, 20, 10)

record_button = st.button("🎙️ Start Recording")
 
if record_button:

    recorded_file = record_audio(duration=duration)

    st.info("⏳ Transcribing your speech...")

    transcription = transcribe_audio(recorded_file)

    st.subheader("🗣️ Transcription:")

    st.write(transcription)
 
    st.info("🔍 Analyzing your speech...")

    results = analyze_transcription(transcription)
 
    st.subheader("📊 IELTS-style Scores (0–9 scale)")

    st.write(f"**Fluency & Coherence:** {results['fluency']:.1f}/9")

    st.write(f"**Grammar & Accuracy:** {results['grammar_score']:.1f}/9")

    st.write(f"**Vocabulary:** {results['vocabulary_richness']:.1f}/9")

    st.write(f"**Overall Score:** {results['total_score']:.1f}/9")
 
    st.subheader("📈 Statistics")

    st.write(f"- Words spoken: {results['word_count']}")

    st.write(f"- Sentences: {results['sentence_count']}")
 
    st.subheader("💬 Feedback")

    st.info(results['feedback'])
 
    # Playback button

    if st.button("🔊 Play Back Transcription"):

        audio_file = text_to_speech(transcription)

        if audio_file:

            with open(audio_file, "rb") as f:

                audio_bytes = f.read()

                st.audio(audio_bytes, format="audio/mp3")

 
