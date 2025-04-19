from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

wakeup_classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

STT = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small.en", device=device
)

processor_TTS = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_TTS = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder_TTS = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)