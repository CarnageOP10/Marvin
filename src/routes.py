from src import app
from src.utils import wakeup_fn, transcribe
from src.models import wakeup_classifier, STT, processor_TTS, model_TTS
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch
from flask import render_template, redirect, url_for, flash, request

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/help')
def help():
    state = False
    while(state == False):
        state = wakeup_fn()
        if state == True:
            flash("Wake word detected!")
        else:
            flash("Listening for wake word...")

    while True:
        text = transcribe()
        if text:
            flash(f"Transcribed text: {text}")
            break
        else:
            flash("No speech detected. Please try again.")

    return render_template('help.html')