import torch
import torchaudio
import os
import sys
import argparse
import subprocess
import requests
import tempfile
from sam_audio import SAMAudio, SAMAudioProcessor

def run_separation():
    parser = argparse.ArgumentParser(description="SAM Audio Universal Separation Tool")
    parser.add_argument("--input", required=True, help="Path locale o URL (mp3, wav, mp4)")
    parser.add_argument("--prompt", required=True, help="Descrizione audio (es. 'a man speaking')")
    parser.add_argument("--output", default="output_final.wav", help="File di output")
    args = parser.parse_args()

    # --- SETUP GPU (Blackwell GB10) ---
    if not torch.cuda.is_available():
        sys.exit("❌ CUDA non rilevata.")
    device = torch.device("cuda")
    
    # Pulizia memoria e disabilitazione TF32 per stabilità segnale
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        # 1. Caricamento Modello e Processor
        print(f"[*] Caricamento modello sulla GPU {torch.cuda.get_device_name(0)}...")
        model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
        processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

        # 2. Gestione Input (Download se URL)
        actual_input = args.input
        tmp_download = None
        if args.input.startswith("http"):
            print("[*] Download file remoto...")
            r = requests.get(args.input)
            tmp_download = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            tmp_download.write(r.content)
            tmp_download.close()
            actual_input = tmp_download.name

        # 3. STANDARDIZZAZIONE IN MP4 (Il formato che funziona sulla tua macchina)
        # Creiamo un video dummy per attivare il percorso di decodifica stabile
        temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        print(f"[*] Preparazione container MP4 compatibile...")
        
        # FFmpeg: video nero + audio AAC 48kHz Mono (lo standard 'Office.mp4')
        cmd = [
            "ffmpeg", "-y", "-i", actual_input,
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1", # Video minimo per non pesare
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-ar", "48000", "-ac", "1",
            "-shortest", "-pix_fmt", "yuv420p",
            temp_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Inferenza (Stessa logica del file funzionante)
        print(f"[*] Elaborazione prompt: '{args.prompt}'...")
        inputs = processor(audios=[temp_mp4], descriptions=[args.prompt]).to(device)
        
        with torch.inference_mode():
            result = model.separate(inputs)

        # 5. Recupero, Normalizzazione e Salvataggio PCM 16-bit
        audio_tensor = result.target[0].cpu().float()
        
        # Rimuoviamo il silenzio digitale (DC Offset)
        audio_tensor = audio_tensor - audio_tensor.mean()
        
        max_val = audio_tensor.abs().max().item()
        print(f"[*] Diagnosi finale: Picco={max_val:.4f}")

        if max_val > 0:
            audio_tensor = audio_tensor / max_val * 0.9
        
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Salvataggio nel formato più compatibile esistente
        torchaudio.save(
            args.output, 
            audio_tensor, 
            processor.audio_sampling_rate, 
            encoding="PCM_S", 
            bits_per_sample=16
        )
        
        print(f"[✅] SUCCESSO: {args.output}")

    except Exception as e:
        print(f"❌ Errore critico: {e}")
    finally:
        # Pulizia rigorosa file temporanei
        if 'temp_mp4' in locals() and os.path.exists(temp_mp4): os.remove(temp_mp4)
        if tmp_download and os.path.exists(tmp_download.name): os.remove(tmp_download.name)

if __name__ == "__main__":
    run_separation()