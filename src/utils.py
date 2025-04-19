from src.models import wakeup_classifier, STT
import pyaudio
import numpy as np
import sys

def get_pyaudio_stream(sampling_rate, chunk_length_s=2.0, stream_chunk_s=0.25):
    """Returns a generator that yields audio chunks using PyAudio."""
    p = pyaudio.PyAudio()
    chunk_size = int(stream_chunk_s * sampling_rate)
    total_samples = int(chunk_length_s * sampling_rate)

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sampling_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    try:
        while True:
            audio_chunk = []
            remaining_samples = total_samples

            while remaining_samples > 0:
                chunk = stream.read(min(chunk_size, remaining_samples))
                audio_chunk.append(np.frombuffer(chunk, dtype=np.float32))
                remaining_samples -= len(audio_chunk[-1])

            yield np.concatenate(audio_chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def wakeup_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in wakeup_classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {wakeup_classifier.model.config.label2id.keys()}."
        )

    sampling_rate = wakeup_classifier.feature_extractor.sampling_rate

    mic = get_pyaudio_stream(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in wakeup_classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                print(f"Wake word detected: {prediction['label']}")
                return True

def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = STT.feature_extractor.sampling_rate

    mic = get_pyaudio_stream(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in STT(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break

    return item["text"]