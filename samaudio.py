import torch
import torchaudio
import requests
import os
import sys
import argparse
import tempfile
from pathlib import Path
from sam_audio import SAMAudio, SAMAudioProcessor

def run_separation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path locale o URL (mp3/mp4)")
    parser.add_argument("--prompt", required=True, help="Prompt")
    parser.add_argument("--output", default="output.wav", help="Output")
    parser.add_argument("--model", default="facebook/sam-audio-large")
    args = parser.parse_args()

    # --- GPU STRICT ---
    if not torch.cuda.is_available():
        print("❌ ERRORE: CUDA non trovata. Esco.")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"[*] Utilizzo GPU: {torch.cuda.get_device_name(0)}")

    try:
        # 1. Caricamento Modello
        print(f"[*] Caricamento modello {args.model}...")
        model = SAMAudio.from_pretrained(args.model).to(device).eval()
        processor = SAMAudioProcessor.from_pretrained(args.model)
        target_sr = processor.audio_sampling_rate

        # 2. Gestione Input (Download se URL)
        input_path = args.input
        if input_path.startswith("http"):
            print("[*] Download file da URL...")
            r = requests.get(input_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(input_path).suffix) as f:
                f.write(r.content)
                input_path = f.name

        # 3. PRE-PROCESSING MANUALE (La chiave per risolvere il silenzio dell'MP3)
        # Carichiamo con torchaudio che è molto più potente del loader interno
        print(f"[*] Pre-elaborazione audio...")
        waveform, sr = torchaudio.load(input_path)
        
        # Mixdown a Mono se Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resampling alla frequenza del modello
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Normalizzazione picco a 0.9 per evitare clipping o segnali troppo deboli
        waveform = waveform / (waveform.abs().max() + 1e-8) * 0.9
        
        # SALVATAGGIO IN WAV TEMPORANEO "PULITO"
        # Questo garantisce che il processor legga un file che capisce al 100%
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(temp_wav.name, waveform, target_sr)
        temp_wav.close()
        
        print(f"[+] Audio normalizzato e pronto ({target_sr}Hz, Mono)")

        # 4. Inferenza
        print(f"[*] Separazione con prompt: '{args.prompt}'...")
        # Passiamo il path del WAV temporaneo
        inputs = processor(audios=[temp_wav.name], descriptions=[args.prompt]).to(device)
        
        with torch.inference_mode():
            result = model.separate(inputs)

        # 5. Salvataggio Finale
        output_tensor = result.target[0].cpu()
        if output_tensor.ndim == 1:
            output_tensor = output_tensor.unsqueeze(0)

        # Controllo energia finale
        energy = output_tensor.abs().max().item()
        if energy < 1e-5:
            print("⚠️ ATTENZIONE: Il modello ha restituito silenzio totale.")
            print("Sperimenta con un prompt più semplice (es. 'human voice' invece di 'a man speaking').")
        else:
            print(f"[+] Segnale generato (Ampiezza picco: {energy:.4f})")

        torchaudio.save(args.output, output_tensor, target_sr)
        print(f"[✅] COMPLETATO: {args.output}")

        # Pulizia
        if os.path.exists(temp_wav.name): os.remove(temp_wav.name)
        if "tempfile" in input_path: os.remove(input_path)

    except Exception as e:
        print(f"❌ Errore critico: {e}")
        raise e

if __name__ == "__main__":
    run_separation()