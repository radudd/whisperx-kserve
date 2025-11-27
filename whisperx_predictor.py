import whisperx
import torch
import kserve
from typing import Dict
import tempfile
import os
import base64
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperXPredictor(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.device = None
        self.compute_type = None
        
    def load(self):
        """Load models on startup"""
        # Determine device and appropriate compute type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set compute type based on device
        if self.device == "cuda":
            self.compute_type = "float16"
            logger.info("Using GPU with float16 compute type")
        else:
            self.compute_type = "int8"  # CPU-friendly compute type
            logger.info("Using CPU with int8 compute type")
        
        model_name = os.getenv("WHISPER_MODEL", "base")  # Use 'base' as default for CPU
        batch_size = int(os.getenv("BATCH_SIZE", "8"))
        
        logger.info(f"Loading Whisper model: {model_name} on {self.device}")
        
        # Load Whisper model with appropriate settings
        self.whisper_model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=self.compute_type,
            download_root=os.getenv("HF_HOME", "/opt/app-root/src/.cache/huggingface")
        )
        
        logger.info(f"Model loaded successfully: {model_name}")
        self.ready = True
    
    def preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Preprocess the input payload
        Expects: {"instances": [{"audio": "base64_encoded_audio", "align": true, "diarize": false}]}
        """
        instances = payload.get("instances", [])
        if not instances:
            raise ValueError("No instances provided in payload")
        
        return instances[0]  # Process single instance
    
    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Perform inference with WhisperX pipeline
        """
        logger.info("Starting inference...")
        
        # Extract parameters
        audio_base64 = request.get("audio")
        align = request.get("align", True)
        diarize = request.get("diarize", False)
        min_speakers = request.get("min_speakers")
        max_speakers = request.get("max_speakers")
        language = request.get("language")  # Optional: specify language
        
        if not audio_base64:
            raise ValueError("No audio data provided")
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            # Check if it looks like a filename instead of base64
            if isinstance(audio_base64, str) and (audio_base64.endswith('.wav') or audio_base64.endswith('.mp3') or audio_base64.endswith('.m4a')):
                raise ValueError(
                    f"Audio field appears to be a filename ('{audio_base64}') instead of base64-encoded audio data. "
                    f"Please encode your audio file to base64 first. "
                    f"Example: base64 -i test.wav | tr -d '\n'"
                )
            else:
                raise ValueError(f"Invalid base64 audio data: {e}. Make sure the audio field contains base64-encoded audio bytes.")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # 1. Load audio
            logger.info("Loading audio...")
            audio = whisperx.load_audio(tmp_path)
            
            # 2. Transcribe with Whisper
            batch_size = int(os.getenv("BATCH_SIZE", "8"))
            logger.info(f"Transcribing with batch_size={batch_size}...")
            
            transcribe_options = {
                "audio": audio,
                "batch_size": batch_size
            }
            if language:
                transcribe_options["language"] = language
                
            result = self.whisper_model.transcribe(**transcribe_options)
            
            language_code = result["language"]
            segments = result["segments"]
            logger.info(f"Transcription complete. Detected language: {language_code}")
            
            # 3. Align with phoneme model (forced alignment)
            word_segments = None
            if align and len(segments) > 0:
                logger.info(f"Loading alignment model for language: {language_code}")
                
                # Load alignment model for detected language if not already loaded
                if self.align_model is None or self.align_metadata is None:
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=self.device
                    )
                
                logger.info("Performing forced alignment...")
                # Perform forced alignment
                result = whisperx.align(
                    segments,
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
                
                segments = result["segments"]
                word_segments = result.get("word_segments", [])
                logger.info("Alignment complete")
            
            # 4. Diarization (optional)
            if diarize:
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    logger.info("Performing speaker diarization...")
                    if self.diarize_model is None:
                        self.diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=hf_token,
                            device=self.device
                        )
                    
                    # Perform diarization
                    diarize_result = self.diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    
                    # Assign speaker labels
                    result = whisperx.assign_word_speakers(diarize_result, result)
                    segments = result["segments"]
                    logger.info("Diarization complete")
                else:
                    logger.warning("HF_TOKEN not set, skipping diarization")
            
            # Clean up memory
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Inference complete")
            return {
                "language": language_code,
                "segments": segments,
                "word_segments": word_segments
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise
        finally:
            os.unlink(tmp_path)
    
    def postprocess(self, result: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Postprocess the prediction result into KServe response format
        """
        return {"predictions": [result]}


if __name__ == "__main__":
    import sys
    
    # Set model name from environment or use default
    model_name = os.getenv("MODEL_NAME", "whisperx")
    
    logger.info(f"Starting KServe ModelServer with model: {model_name}")
    
    # Create and load model
    model = WhisperXPredictor(model_name)
    model.load()
    
    # Start KServe server
    try:
        kserve.ModelServer().start([model])
    except Exception as e:
        logger.error(f"Failed to start ModelServer: {e}")
        sys.exit(1)