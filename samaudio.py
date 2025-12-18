import torch
from IPython.display import Audio, Video

from sam_audio import SAMAudio, SAMAudioProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SAMAudio.from_pretrained("facebook/sam-audio-3b").to(device).eval()
model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
#processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-3b")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
video_file = "/gorgon/ia/sam-audio/assets/office.mp4"
Video(video_file, embed=True, width=640, height=360)
inputs = processor(audios=[video_file], descriptions=["A man speaking"]).to(device)
with torch.inference_mode():
    result = model.separate(inputs)
Audio(result.target[0].cpu(), rate=processor.audio_sampling_rate)