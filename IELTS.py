import os
import io
import tkinter as tk
import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import speech, texttospeech
import spacy
from collections import Counter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/happy/OneDrive/Documents/Projects/ielts-speaking-test-448518-cd07b827ad03.json"

nlp = spacy.load("en_core_web_sm")


def record_audio(filename, duration=10, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio_data)
    print(f"Recording saved to {filename}")


def transcribe_audio(file_path):
    client = speech.SpeechClient()
    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-ZA"
    )

    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        return result.alternatives[0].transcript


def text_to_speech(text, output_file):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-ZA",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to {output_file}")

# Analyze transcription


def analyze_transcription(transcription):
    doc = nlp(transcription)
    word_count = len(doc)
    pos_counts = Counter([token.pos_ for token in doc])
    grammar_score = pos_counts.get("VERB", 0) + pos_counts.get("NOUN", 0)
    unique_words = len(set([token.text.lower()
                       for token in doc if token.is_alpha]))
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0
    total_score = (word_count * 0.4) + (grammar_score * 0.3) + \
        (vocabulary_richness * 0.3)

    print(f"Fluency (Word Count): {word_count}")
    print(f"Grammar Score (POS Counts): {grammar_score}")
    print(f"Vocabulary Richness (Unique Words): {vocabulary_richness:.2f}")
    print(f"Total Score: {total_score:.2f}")

    return {
        "fluency": word_count,
        "grammar_score": grammar_score,
        "vocabulary_richness": vocabulary_richness,
        "total_score": total_score
    }


class IELTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IELTS Speaking Test")

        self.transcription_label = tk.Label(
            root, text="Transcription will appear here.", wraplength=300)
        self.transcription_label.pack(pady=10)

        self.record_button = tk.Button(
            root, text="Record", command=self.record_and_transcribe)
        self.record_button.pack(pady=10)

        self.feedback_label = tk.Label(
            root, text="Feedback and analysis will appear here.", wraplength=300)
        self.feedback_label.pack(pady=10)

        self.play_button = tk.Button(
            root, text="Play Transcription", command=self.play_transcription)
        self.play_button.pack(pady=10)

        self.output_audio_file = "output_audio.mp3"

    def record_and_transcribe(self):
        recorded_file = "recorded_audio.wav"
        record_audio(recorded_file, duration=10)

        transcription = transcribe_audio(recorded_file)
        self.transcription_label.config(text=f"Transcription: {transcription}")

        analysis = analyze_transcription(transcription)
        feedback = (
            f"Fluency (Word Count): {analysis['fluency']}\n"
            f"Grammar Score: {analysis['grammar_score']}\n"
            f"Vocabulary Richness: {analysis['vocabulary_richness']:.2f}\n"
            f"Total Score: {analysis['total_score']:.2f}"
        )
        self.feedback_label.config(text=feedback)

    def play_transcription(self):
        transcription_text = self.transcription_label.cget(
            "text").replace("Transcription: ", "")
        text_to_speech(transcription_text, self.output_audio_file)
        os.system(f"start {self.output_audio_file}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IELTSApp(root)
    root.mainloop()
