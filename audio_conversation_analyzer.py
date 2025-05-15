import os
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import whisper
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, pipeline
from transformers import pipeline
from pydub import AudioSegment

class AudioConversationAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Conversation Analyzer")
        self.root.geometry("900x700")
        
        # Set up UI components
        self.setup_ui()
        
        # Initialize models
        self.load_models()
        
        # Set up variables
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.transcription = None
        self.num_speakers = 0
        self.separated_speakers = []
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Audio File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Analyze", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_label = ttk.Label(self.progress_frame, text="Status: Ready")
        self.progress_label.pack(side=tk.TOP, anchor=tk.W)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        
        # Notebook for results
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Transcription tab
        self.transcription_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.transcription_frame, text="Transcription")
        
        self.transcription_text = scrolledtext.ScrolledText(self.transcription_frame, wrap=tk.WORD)
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_models(self):
        self.update_status("Loading models... This may take a minute", 10)
        
        # We'll load models on demand to save memory and startup time
        self.whisper_model = None
        self.speaker_separator = None
        self.sentiment_analyzer = None
        self.summarizer = None
        
        self.update_status("Models ready to load on demand", 20)
    
    def browse_file(self):
        filetypes = (
            ('Audio files', '*.wav;*.mp3'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Open an audio file',
            initialdir='/',
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.audio_file = filename
    
    def start_analysis(self):
        if not self.file_path_var.get():
            return
            
        # Start analysis in a separate thread to keep UI responsive
        analysis_thread = threading.Thread(target=self.analyze_audio)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def analyze_audio(self):
        try:
            total_start = time.time()
            self.update_status("Loading audio file...", 20)
            
            # Load audio file
            file_path = self.file_path_var.get()
            
            # Convert mp3 to wav if needed
            if file_path.lower().endswith('.mp3'):
                self.update_status("Converting MP3 to WAV...", 25)
                sound = AudioSegment.from_mp3(file_path)
                temp_wav = file_path.replace('.mp3', '.wav')
                if not os.path.exists(temp_wav):
                    sound.export(temp_wav, format="wav")
                file_path = temp_wav
            
            # Load audio data
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=True)
            
            # Display waveform
            self.update_status("Generating waveform visualization...", 30)
            self.display_waveform()
            
            self.update_status("Detecting speakers and transcribing audio...", 40)

            # Timing for detection and transcription
            self.speaker_time = 0
            self.transcribe_time = 0

            def timed_detect():
                start = time.time()
                self.detect_speakers()
                self.speaker_time = time.time() - start

            def timed_transcribe():
                start = time.time()
                self.transcribe_audio()
                self.transcribe_time = time.time() - start

            speaker_thread = threading.Thread(target=timed_detect)
            transcribe_thread = threading.Thread(target=timed_transcribe)

            speaker_thread.start()
            transcribe_thread.start()
            speaker_thread.join()
            transcribe_thread.join()

            self.analysis_time = time.time() - total_start

            # Analyze sentiment
            self.update_status("Analyzing sentiment...", 80)
            sentiment = self.analyze_sentiment()
            
            # Generate summary
            self.update_status("Generating summary...", 90)
            summary = self.generate_summary()
            
            # Display results
            self.update_status("Displaying results...", 95)
            self.display_results(sentiment, summary)
            
            self.update_status("Analysis complete!", 100)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 0)
            import traceback
            print(traceback.format_exc())
    
    def update_status(self, message, progress):
        def update():
            self.progress_label.config(text=f"Status: {message}")
            self.progress_var.set(progress)
            self.root.update_idletasks()
        
        self.root.after(0, update)
    
    def display_waveform(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Time axis
        time = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
        
        # Plot waveform
        ax.plot(time, self.audio_data, color='blue', alpha=0.7)
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        
        self.canvas.draw()
    
    def detect_speakers(self):
        # Method 1: Using energy-based VAD and clustering
        # For basic estimation, we'll use a simple energy-based approach
        
        # Calculate energy
        energy = np.square(self.audio_data)
        
        # Moving average
        window_size = int(0.5 * self.sample_rate)  # 500ms window
        energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        
        # Threshold
        threshold = 0.05 * np.max(energy_smooth)
        voice_activity = energy_smooth > threshold
        
        # Count segments (crude speaker estimation)
        segments = np.where(np.diff(voice_activity.astype(int)))[0]
        
        # Assuming turn-taking conversation
        estimated_speakers = min(max(1, len(segments) // 4), 6)  # Reasonable bounds
        
        # For a more accurate approach, we would load a speaker diarization model
        # But for simplicity, we'll use this estimation
        self.num_speakers = estimated_speakers
    
    def load_whisper_model(self):
        if self.whisper_model is None:
            self.update_status("Loading Whisper transcription model...", 40)
            self.whisper_model = whisper.load_model("turbo")  # Using turbo model for efficiency
    
    def transcribe_audio(self):
        self.load_whisper_model()
        
        try:
            # First attempt - use whisper's default method
            result = self.whisper_model.transcribe(self.file_path_var.get(),language="he")
            self.transcription = result["text"]
        except FileNotFoundError:
            # If FFmpeg is not found, use librosa to load the audio instead
            self.update_status("FFmpeg not found. Using alternative transcription method...", 65)
            
            # Load audio using librosa
            audio, _ = librosa.load(self.file_path_var.get(), sr=16000)
            
            # Ensure audio is in the correct format for Whisper (float32 numpy array)
            audio = audio.astype(np.float32)
            
            # Directly transcribe the loaded audio
            result = self.whisper_model.transcribe(audio,language="he")
            self.transcription = result["text"]
            
        # Update transcription text area
        def update_text():
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, self.transcription)
        
        self.root.after(0, update_text)
    
    def load_sentiment_model(self):
        if self.sentiment_analyzer is None:
            self.update_status("Loading sentiment analysis model...", 70)
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def analyze_sentiment(self):
        self.load_sentiment_model()
        
        # Break text into chunks if needed (transformers have max token limits)
        words = self.transcription.split()
        chunks = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
        
        sentiments = []
        for chunk in chunks:
            if chunk.strip():
                sentiment = self.sentiment_analyzer(chunk)[0]
                sentiments.append(sentiment)
        
        # Aggregate sentiment
        pos_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        neg_count = len(sentiments) - pos_count
        
        if pos_count > neg_count:
            overall_sentiment = "Positive/Interested"
        elif neg_count > pos_count:
            overall_sentiment = "Negative/Frustrated"
        else:
            overall_sentiment = "Neutral"
        
        return overall_sentiment
    
    def load_summarizer(self):
        if self.summarizer is None:
            self.update_status("Loading summarization model...", 85)
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            # Set Hebrew as the target language
            tokenizer.src_lang = "he_IL"
            tokenizer.tgt_lang = "he_IL"
            self.summarizer_tokenizer = tokenizer
            self.summarizer_model = model
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer
            )
    
    def generate_summary(self):
        self.load_summarizer()
        
        # Check if text is long enough to summarize
        if len(self.transcription.split()) < 10:
            return "Text too short to summarize."
        
        # Break into chunks if needed
        words = self.transcription.split()
        chunks = [' '.join(words[i:i+500]) for i in range(0, len(words), 500)]
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            print("Summarizing chunk:", chunk)
            if len(chunk.split()) > 10:
                self.summarizer.tokenizer.src_lang = "he_IL"
                hebrew_token_id = self.summarizer.tokenizer.lang_code_to_id["he_IL"]
                self.summarizer.tokenizer.tgt_lang = "he_IL"# Only summarize if enough text
                summary = self.summarizer(
                                        chunk,
                                        max_length=100,
                                        min_length=30,
                                        do_sample=False,
                                        forced_bos_token_id=hebrew_token_id
                                    )
                summaries.append(summary[0]['summary_text'])
        
        if not summaries:
            return "Could not generate summary."
        print(" ".join(summaries))
        return " ".join(summaries)
    
    def display_results(self, sentiment, summary):
        analysis_text = f"""
Audio Analysis Results:
======================

Number of participants detected: {self.num_speakers}

Time to detect speakers: {self.speaker_time:.2f} seconds
Time to transcribe: {self.transcribe_time:.2f} seconds
Total analysis time: {self.analysis_time:.2f} seconds

Sentiment Analysis:
------------------
Overall tone: {sentiment}

Conversation Summary:
-------------------
{summary}
"""
        def update_analysis():
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, analysis_text)
            self.notebook.select(self.analysis_frame)  # Switch to analysis tab
        
        self.root.after(0, update_analysis)


def main():
    root = tk.Tk()
    app = AudioConversationAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()