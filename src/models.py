from transformers import AutoProcessor, AutoModelForTextToSpectrogram, pipeline
import torch 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

wakeup_classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

STT = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small.en", device=device
)

processor_TTS = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
model_TTS = AutoModelForTextToSpectrogram.from_pretrained("microsoft/speecht5_tts")