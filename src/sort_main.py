import torch
from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACE_API_KEY= os.getenv("HUGGINGFACE_API_KEY")


from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# inference on the whole file
pipeline("../data/1.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)

from pyannote.audio import Audio
waveform, sample_rate = Audio().crop("1_output.wav", excerpt)
pipeline({"waveform": waveform, "sample_rate": sample_rate})