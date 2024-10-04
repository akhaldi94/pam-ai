import pkgutil
import re
from pathlib import Path
import numpy as np
import pyaudio
import soundfile as sf
import tempfile

# patch whisper on file not find error
# https://github.com/carloscdias/whisper-cpp-python/pull/12
try:
    from whisper_cpp_python import Whisper
except FileNotFoundError:
    regex = r"(\"darwin\":\n\s*lib_ext = \")\.so(\")"
    subst = "\\1.dylib\\2"

    print("fixing and re-importing whisper_cpp_python...")
    # load whisper_cpp_python and substitute .so with .dylib for darwin
    package = pkgutil.get_loader("whisper_cpp_python")
    whisper_path = Path(package.path)
    whisper_cpp_py = whisper_path.parent.joinpath("whisper_cpp.py")
    content = whisper_cpp_py.read_text()
    result = re.sub(regex, subst, content, 0, re.MULTILINE)
    whisper_cpp_py.write_text(result)

    from whisper_cpp_python import Whisper
from utils.yaml_extraction import get_model_path

# Constants for audio capture
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono channel
RATE = 16000  # Sampling rate
CHUNK = 8096  # Number of audio frames per buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

model_path = get_model_path("../../variables.yaml")
print(model_path)

model = Whisper("/Users/kahmed/Projects/whisper.cpp/models/ggml-medium.bin")

print("Recording...")

try:
    while True:
        # Read audio data from the microphone
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Ensure the audio data is in the correct shape
        if audio_data.shape[0] % CHUNK != 0:
            audio_data = np.pad(audio_data, (0, CHUNK - audio_data.shape[0] % CHUNK), mode='constant')

        # Write audio_data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            sf.write(temp_wav.name, audio_data, RATE)
            transcription = model.transcribe(temp_wav.name)

        # Print the transcription
        print("Transcription:", transcription)

except KeyboardInterrupt:
    print("Recording stopped.")

finally:
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()