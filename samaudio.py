import torch
from IPython.display import Audio, Video
import torchaudio

from sam_audio import SAMAudio, SAMAudioProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SAMAudio.from_pretrained("facebook/sam-audio-3b").to(device).eval()
model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
#processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-3b")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
video_file = "/gorgon/ia/sam-audio/examples/assets/office.mp4"
Video(video_file, embed=True, width=640, height=360)
inputs = processor(audios=[video_file], descriptions=["A man speaking"]).to(device)
with torch.inference_mode():
    result = model.separate(inputs)
# Audio(result.target[0].cpu(), rate=processor.audio_sampling_rate)

# 1. Recupera il tensore dell'audio (target[0]) e portalo sulla CPU
audio_tensor = result.target[0].cpu()

# 2. Torchaudio si aspetta un tensore con forma [canali, campioni]
# Se il tensore è 1D (mono), aggiungiamo una dimensione
if audio_tensor.ndim == 1:
    audio_tensor = audio_tensor.unsqueeze(0)

# 3. Definisci il nome del file e salva
output_filename = "audio_separato.wav"
sample_rate = processor.audio_sampling_rate

torchaudio.save(output_filename, audio_tensor, sample_rate)

print(f"File salvato con successo: {output_filename}")

# Opzionale: continua a visualizzare il player nel notebook
Audio(audio_tensor.numpy(), rate=sample_rate)