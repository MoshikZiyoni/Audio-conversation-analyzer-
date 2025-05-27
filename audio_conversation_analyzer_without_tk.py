import os
import time
import numpy as np
import threading
import librosa
import whisper
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, pipeline
from pydub import AudioSegment
import argparse # Import argparse for command-line arguments

class AudioConversationAnalyzer:
    def __init__(self):
        # Initialize models (no need for 'root' or UI setup)
        self.load_models()

        # Set up variables
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.transcription = None
        self.num_speakers = 0
        self.separated_speakers = []

        # Variables for timing
        self.speaker_time = 0
        self.transcribe_time = 0
        self.analysis_time = 0

    def load_models(self):
        # In a CLI, we print status to console instead of updating a GUI label
        print("Status: Loading models... This may take a minute")

        # We'll load models on demand to save memory and startup time
        self.whisper_model = None
        self.speaker_separator = None
        self.sentiment_analyzer = None
        self.summarizer = None

        print("Status: Models ready to load on demand")

    def analyze_audio_cli(self, file_path):
        """
        Main method to analyze an audio file via CLI.
        """
        self.audio_file = file_path # Set the audio file path
        
        try:
            total_start = time.time()
            print(f"Status: Loading audio file: {file_path}...")

            # Convert mp3 to wav if needed using pydub
            if file_path.lower().endswith('.mp3'):
                print("Status: Converting MP3 to WAV...")
                # Create a temporary WAV file in the same directory as the input MP3
                temp_wav = os.path.splitext(file_path)[0] + '.wav'
                
                # Check if the temporary WAV already exists to avoid re-conversion
                if not os.path.exists(temp_wav):
                    sound = AudioSegment.from_mp3(file_path)
                    sound.export(temp_wav, format="wav")
                
                file_path = temp_wav # Use the WAV for further processing
                print(f"Status: MP3 converted to WAV: {file_path}")
            
            # Load audio data using librosa
            # This is robust and handles various formats if ffmpeg is installed
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=True)
            print("Status: Audio loaded successfully.")

            # In a CLI, we don't display a waveform, but we can print its info
            print("Status: Audio data loaded for analysis.")
            
            print("Status: Detecting speakers and transcribing audio...")

            # Run detection and transcription in parallel threads
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
            print("Status: Analyzing sentiment...")
            sentiment = self.analyze_sentiment()
            
            # Generate summary
            print("Status: Generating summary...")
            summary = self.generate_summary()
            
            # Display results to console
            print("Status: Displaying results...")
            self.print_results(sentiment, summary)
            
            print("Status: Analysis complete!")
            
        except FileNotFoundError:
            print(f"Error: Audio file not found at '{file_path}'. Please check the path.")
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            print(traceback.format_exc())
            
    # The update_status and display_waveform methods are GUI-specific and removed.
    # The browse_file and start_analysis methods are also GUI-specific and removed.

    def detect_speakers(self):
        # Method 1: Using energy-based VAD and clustering
        # For basic estimation, we'll use a simple energy-based approach
        
        # Calculate energy
        energy = np.square(self.audio_data)
        
        # Moving average
        window_size = int(0.5 * self.sample_rate)   # 500ms window
        energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        
        # Threshold
        threshold = 0.05 * np.max(energy_smooth)
        voice_activity = energy_smooth > threshold
        
        # Count segments (crude speaker estimation)
        segments = np.where(np.diff(voice_activity.astype(int)))[0]
        
        # Assuming turn-taking conversation
        estimated_speakers = min(max(1, len(segments) // 4), 6)   # Reasonable bounds
        
        self.num_speakers = estimated_speakers
    
    def load_whisper_model(self):
        if self.whisper_model is None:
            print("Status: Loading Whisper transcription model (turbo version)...")
            self.whisper_model = whisper.load_model("turbo") # Using turbo model for efficiency
            print("Status: Whisper model loaded.")
    
    def transcribe_audio(self):
        self.load_whisper_model()
        
        try:
            # Whisper's transcribe method generally handles file paths and loads audio itself.
            # The pydub RuntimeWarning you saw earlier is from pydub, not Whisper directly.
            # Whisper internally might use ffmpeg if available.
            # If ffmpeg is NOT available, it falls back to soundfile, which needs to be installed.
            # Ensure soundfile is in your requirements.txt.
            
            # Use self.audio_file, which is the path to the original or temp WAV file
            result = self.whisper_model.transcribe(self.audio_file, language="he")
            self.transcription = result["text"]
            print("Status: Transcription complete.")

        except Exception as e:
            # Catching a broader exception here as FileNotFoundError might not be the only issue
            print(f"Warning: Could not transcribe directly with Whisper (likely due to missing FFmpeg or other audio issue). Trying alternative loading method.")
            print(f"Error details: {e}") # Print error for debugging
            
            # Fallback using librosa to load audio explicitly before passing to Whisper
            print("Status: Using librosa to load audio for transcription fallback...")
            try:
                audio, _ = librosa.load(self.audio_file, sr=16000) # Whisper expects 16kHz
                audio = audio.astype(np.float32) # Ensure correct data type
                
                result = self.whisper_model.transcribe(audio, language="he")
                self.transcription = result["text"]
                print("Status: Transcription fallback complete via librosa.")
            except Exception as librosa_e:
                print(f"Error: Failed to load audio with librosa for transcription fallback: {librosa_e}")
                self.transcription = "Transcription failed."

    def load_sentiment_model(self):
        if self.sentiment_analyzer is None:
            print("Status: Loading sentiment analysis model...")
            # Using a general sentiment-analysis pipeline. You can specify a model
            # like "cardiffnlp/twitter-roberta-base-sentiment" if you need a specific one.
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            print("Status: Sentiment analysis model loaded.")
    
    def analyze_sentiment(self):
        self.load_sentiment_model()
        
        if not self.transcription:
            return "No transcription available for sentiment analysis."

        # Break text into chunks if needed (transformers have max token limits)
        words = self.transcription.split()
        chunks = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)] # Smaller chunks for general models
        
        sentiments = []
        for chunk in chunks:
            if chunk.strip():
                try:
                    sentiment = self.sentiment_analyzer(chunk)[0]
                    sentiments.append(sentiment)
                except Exception as e:
                    print(f"Warning: Could not analyze sentiment for chunk: {chunk[:50]}... Error: {e}")
        
        if not sentiments:
            return "Could not determine sentiment (no valid chunks analyzed)."

        # Aggregate sentiment
        pos_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        neg_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE') # Also count explicit negative
        
        if pos_count > neg_count:
            overall_sentiment = "Positive/Interested"
        elif neg_count > pos_count:
            overall_sentiment = "Negative/Frustrated"
        else:
            overall_sentiment = "Neutral"
        
        return overall_sentiment
    
    def load_summarizer(self):
        if self.summarizer is None:
            print("Status: Loading summarization model...")
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            
            # Set Hebrew as the target language for the tokenizer
            tokenizer.src_lang = "he_IL"
            tokenizer.tgt_lang = "he_IL" # For generation as well
            
            self.summarizer_tokenizer = tokenizer
            self.summarizer_model = model
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                # Add device if you have a GPU: device=0 (for cuda) or device=-1 (for cpu)
            )
            print("Status: Summarization model loaded.")
    
    def generate_summary(self):
        self.load_summarizer()
        
        if not self.transcription:
            return "No transcription available for summarization."

        # Check if text is long enough to summarize
        if len(self.transcription.split()) < 10:
            return "Text too short to summarize."
        
        # Break into chunks if needed (MBart has high token limits, but better safe)
        words = self.transcription.split()
        chunks = [' '.join(words[i:i+500]) for i in range(0, len(words), 500)]
        
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Status: Summarizing chunk {i+1}/{len(chunks)}...")
            if len(chunk.split()) > 10: # Only summarize if enough text
                try:
                    # Ensure language settings are consistent for the pipeline
                    self.summarizer.tokenizer.src_lang = "he_IL"
                    hebrew_token_id = self.summarizer.tokenizer.lang_code_to_id["he_IL"]
                    self.summarizer.tokenizer.tgt_lang = "he_IL"
                    
                    summary_output = self.summarizer(
                        chunk,
                        max_length=100,
                        min_length=30,
                        do_sample=False,
                        forced_bos_token_id=hebrew_token_id # Forces generation in Hebrew
                    )
                    summaries.append(summary_output[0]['summary_text'])
                except Exception as e:
                    print(f"Warning: Could not generate summary for chunk {i+1}. Error: {e}")
        
        if not summaries:
            return "Could not generate summary."
            
        final_summary = " ".join(summaries)
        print("Status: Summarization complete.")
        return final_summary
    
    def print_results(self, sentiment, summary):
        """
        Prints the analysis results to the console.
        """
        print("\n" + "="*40)
        print("    Audio Conversation Analysis Results")
        print("="*40 + "\n")

        print("--- Timing ---")
        print(f"Time to detect speakers: {self.speaker_time:.2f} seconds")
        print(f"Time to transcribe: {self.transcribe_time:.2f} seconds")
        print(f"Total analysis time: {self.analysis_time:.2f} seconds\n")
        
        print("--- Transcription ---")
        print(self.transcription if self.transcription else "Transcription not available.\n")

        print("--- Analysis ---")
        print(f"Number of participants detected: {self.num_speakers}")
        print(f"Overall tone: {sentiment}\n")

        print("--- Conversation Summary ---")
        print(summary if summary else "Summary not available.\n")
        print("="*40 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio conversations from the command line."
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file (.mp3 or .wav) to be analyzed."
    )
    args = parser.parse_args()

    # Instantiate the analyzer without Tkinter root
    app = AudioConversationAnalyzer()
    
    # Call the CLI analysis method
    app.analyze_audio_cli(args.audio_file)

if __name__ == "__main__":
    main()