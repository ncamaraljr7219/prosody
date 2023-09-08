import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call

# Function to record audio
def record_audio(seconds=5, sample_rate=44100):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []
    for _ in range(0, int(sample_rate / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    return audio_data, sample_rate

# Extract prosody features
def extract_prosody_features(audio_data, sample_rate):
    snd = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)

    # Extract fundamental frequency (F0) using Praat
    pitch = call(snd, "To Pitch (cc)", 0, 75, 600)
    f0_values = call(pitch, "Get mean", 0, 0, "Hertz")

    # Extract intensity (energy) using Praat
    intensity = call(snd, "To Intensity", 75, 0, "no")
    intensity_values = call(intensity, "Get mean", 0, 0, "energy")

    return f0_values, intensity_values

if __name__ == "__main__":
    audio_data, sample_rate = record_audio()

    f0_values, intensity_values = extract_prosody_features(audio_data, sample_rate)

    # Visualize prosody features (fundamental frequency and intensity)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f0_values)
    plt.title("Fundamental Frequency (F0)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(2, 1, 2)
    plt.plot(intensity_values)
    plt.title("Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (dB)")

    plt.tight_layout()
    plt.show()
