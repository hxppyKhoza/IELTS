import os
import io
import tkinter as tk
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from gtts import gTTS
import spacy
from collections import Counter
import pygame
import tempfile

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def record_audio(filename, duration=10, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio_data)
    print(f"Recording saved to {filename}")

def transcribe_audio(file_path):
    """Transcribe audio using free SpeechRecognition library"""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        
        # Try Google's free web service first
        try:
            transcription = r.recognize_google(audio)
            print(f"Transcription successful: {transcription}")
            return transcription
        except sr.UnknownValueError:
            return "Could not understand the audio clearly. Please try speaking more clearly."
        except sr.RequestError:
            # Fallback to offline recognition
            try:
                transcription = r.recognize_sphinx(audio)
                print(f"Offline transcription: {transcription}")
                return transcription
            except:
                return "Transcription service unavailable. Please check your internet connection."
    
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error processing audio: {str(e)}"

def text_to_speech(text, output_file):
    """Convert text to speech using free gTTS (Google Text-to-Speech)"""
    try:
        # Create TTS object
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        print(f"Audio content written to {output_file}")
        return True
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False

def analyze_transcription(transcription):
    """Analyze transcription for basic IELTS-like metrics"""
    if not transcription or transcription.startswith("Could not") or transcription.startswith("Error"):
        return {
            "fluency": 0,
            "grammar_score": 0,
            "vocabulary_richness": 0,
            "total_score": 0,
            "feedback": "Unable to analyze - transcription failed"
        }
    
    doc = nlp(transcription)
    
    # Basic metrics
    word_count = len([token for token in doc if token.is_alpha])
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    
    # POS analysis
    pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
    
    # Grammar complexity (variety of POS tags)
    grammar_variety = len(pos_counts)
    verb_count = pos_counts.get("VERB", 0)
    noun_count = pos_counts.get("NOUN", 0)
    adj_count = pos_counts.get("ADJ", 0)
    
    # Vocabulary analysis
    unique_words = len(set([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]))
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0
    
    # Fluency metrics
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Scoring (0-9 scale like IELTS)
    fluency_score = min(9, (word_count / 10) + (avg_words_per_sentence / 5))
    grammar_score = min(9, grammar_variety * 0.5 + (verb_count + noun_count) / word_count * 9)
    vocabulary_score = min(9, vocabulary_richness * 9 + (adj_count / word_count) * 3)
    
    total_score = (fluency_score + grammar_score + vocabulary_score) / 3
    
    # Generate feedback
    feedback_parts = []
    if fluency_score < 5:
        feedback_parts.append("Try to speak more fluently with longer sentences.")
    if grammar_score < 5:
        feedback_parts.append("Focus on using varied sentence structures and grammar.")
    if vocabulary_score < 5:
        feedback_parts.append("Try to use more diverse vocabulary and descriptive words.")
    
    if not feedback_parts:
        feedback_parts.append("Good job! Keep practicing to maintain your level.")
    
    feedback = " ".join(feedback_parts)
    
    print(f"Fluency Score: {fluency_score:.1f}")
    print(f"Grammar Score: {grammar_score:.1f}")
    print(f"Vocabulary Score: {vocabulary_score:.1f}")
    print(f"Total Score: {total_score:.1f}")
    
    return {
        "fluency": fluency_score,
        "grammar_score": grammar_score,
        "vocabulary_richness": vocabulary_score,
        "total_score": total_score,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "feedback": feedback
    }

class IELTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IELTS Speaking Test Practice")
        self.root.geometry("500x600")
        
        # Main frame
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="IELTS Speaking Practice", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = tk.Label(main_frame, 
                               text="Click 'Start Recording' and speak for up to 10 seconds.\nYou'll receive feedback on your fluency, grammar, and vocabulary.",
                               wraplength=450, justify=tk.CENTER)
        instructions.pack(pady=(0, 20))
        
        # Recording section
        recording_frame = tk.Frame(main_frame)
        recording_frame.pack(pady=10)
        
        self.record_button = tk.Button(recording_frame, text="Start Recording", 
                                      command=self.record_and_transcribe,
                                      bg="#4CAF50", fg="white", font=("Arial", 12),
                                      padx=20, pady=10)
        self.record_button.pack()
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready to record", 
                                    fg="blue", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Transcription display
        transcription_frame = tk.Frame(main_frame)
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(transcription_frame, text="Your Speech:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.transcription_text = tk.Text(transcription_frame, height=4, wrap=tk.WORD,
                                         font=("Arial", 10), state=tk.DISABLED)
        transcription_scroll = tk.Scrollbar(transcription_frame)
        transcription_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        self.transcription_text.config(yscrollcommand=transcription_scroll.set)
        transcription_scroll.config(command=self.transcription_text.yview)
        
        # Feedback display
        feedback_frame = tk.Frame(main_frame)
        feedback_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(feedback_frame, text="Analysis & Feedback:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.feedback_text = tk.Text(feedback_frame, height=6, wrap=tk.WORD,
                                    font=("Arial", 10), state=tk.DISABLED)
        feedback_scroll = tk.Scrollbar(feedback_frame)
        feedback_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.feedback_text.pack(fill=tk.BOTH, expand=True)
        self.feedback_text.config(yscrollcommand=feedback_scroll.set)
        feedback_scroll.config(command=self.feedback_text.yview)
        
        # Playback section
        playback_frame = tk.Frame(main_frame)
        playback_frame.pack(pady=10)
        
        self.play_button = tk.Button(playback_frame, text="Play Back Recording", 
                                    command=self.play_transcription,
                                    bg="#2196F3", fg="white", font=("Arial", 10),
                                    padx=15, pady=5, state=tk.DISABLED)
        self.play_button.pack()
        
        # Initialize variables
        self.current_transcription = ""
        self.output_audio_file = tempfile.mktemp(suffix=".mp3")
    
    def update_status(self, message, color="black"):
        """Update status label with message and color"""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def update_transcription_display(self, text):
        """Update transcription text widget"""
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(1.0, text)
        self.transcription_text.config(state=tk.DISABLED)
    
    def update_feedback_display(self, analysis):
        """Update feedback text widget with analysis results"""
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        
        feedback_content = f"""SCORES (0-9 scale):
• Fluency & Coherence: {analysis['fluency']:.1f}/9
• Grammar & Accuracy: {analysis['grammar_score']:.1f}/9  
• Vocabulary & Lexical Resource: {analysis['vocabulary_richness']:.1f}/9
• Overall Score: {analysis['total_score']:.1f}/9

STATISTICS:
• Words spoken: {analysis.get('word_count', 0)}
• Sentences: {analysis.get('sentence_count', 0)}

FEEDBACK:
{analysis['feedback']}

TIPS:
• Practice speaking for longer periods to improve fluency
• Use varied vocabulary and sentence structures
• Focus on clear pronunciation and natural rhythm"""
        
        self.feedback_text.insert(1.0, feedback_content)
        self.feedback_text.config(state=tk.DISABLED)
    
    def record_and_transcribe(self):
        """Main function to record audio and provide analysis"""
        try:
            # Disable button during recording
            self.record_button.config(state=tk.DISABLED)
            self.update_status("Recording... Speak now!", "red")
            
            # Record audio
            recorded_file = "recorded_audio.wav"
            record_audio(recorded_file, duration=10)
            
            self.update_status("Processing audio...", "orange")
            
            # Transcribe audio
            transcription = transcribe_audio(recorded_file)
            self.current_transcription = transcription
            self.update_transcription_display(transcription)
            
            self.update_status("Analyzing speech...", "orange")
            
            # Analyze transcription
            analysis = analyze_transcription(transcription)
            self.update_feedback_display(analysis)
            
            # Enable playback button
            self.play_button.config(state=tk.NORMAL)
            
            self.update_status("Analysis complete! You can record again or play back.", "green")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            print(f"Error in record_and_transcribe: {e}")
        
        finally:
            # Re-enable record button
            self.record_button.config(state=tk.NORMAL)
    
    def play_transcription(self):
        """Convert transcription to speech and play it back"""
        if not self.current_transcription or self.current_transcription.startswith(("Could not", "Error")):
            self.update_status("No valid transcription to play back", "red")
            return
        
        try:
            self.update_status("Generating audio...", "orange")
            
            # Generate speech audio
            if text_to_speech(self.current_transcription, self.output_audio_file):
                self.update_status("Playing audio...", "blue")
                
                # Play the audio using pygame
                pygame.mixer.music.load(self.output_audio_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    self.root.update()
                    pygame.time.wait(100)
                
                self.update_status("Playback complete", "green")
            else:
                self.update_status("Failed to generate audio", "red")
                
        except Exception as e:
            self.update_status(f"Playback error: {str(e)}", "red")
            print(f"Error in play_transcription: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IELTSApp(root)
    root.mainloop()
