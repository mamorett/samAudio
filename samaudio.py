import torch
import torchaudio
import gc
import os
import types
import argparse
from sam_audio import SAMAudio, SAMAudioProcessor

def create_lite_model(model_name="facebook/sam-audio-base"):
    """ 
    Carica il modello Base e rimuove i componenti inutilizzati 
    per massimizzare la stabilità su Linux.
    """
    print(f"[*] Caricamento e ottimizzazione di {model_name}...")
    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)

    # Salvataggio dimensione per il dummy encoder
    vision_dim = model.vision_encoder.dim if hasattr(model.vision_encoder, 'dim') else 1024
    
    # Rimozione componenti pesanti
    del model.vision_encoder
    model._vision_encoder_dim = vision_dim

    def _get_video_features_lite(self, video, audio_features):
        B, T, _ = audio_features.shape
        return audio_features.new_zeros(B, self._vision_encoder_dim, T)

    model._get_video_features = types.MethodType(_get_video_features_lite, model)

    for component in ['visual_ranker', 'text_ranker', 'span_predictor', 'span_predictor_transform']:
        if hasattr(model, component):
            delattr(model, component)
            setattr(model, component, None)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, processor

def run_separation():
    parser = argparse.ArgumentParser(description="SAM Audio Base Separation")
    parser.add_argument("--input", "-i", required=True, help="File audio di input")
    parser.add_argument("--prompt", "-p", required=True, help="Prompt testuale")
    parser.add_argument("--output", "-o", default="output_base.wav", help="File di output")
    parser.add_argument("--chunk_sec", type=float, default=20.0, help="Durata chunk (sec)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # bfloat16 è ideale per Linux/NVIDIA per evitare overflow e silenzio
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    try:
        # 1. Setup Modello Base
        model, processor = create_lite_model("facebook/sam-audio-base")
        model = model.to(device, dtype).eval()
        
        # 2. Caricamento e Pre-processing Audio
        sample_rate = processor.audio_sampling_rate
        audio, orig_sr = torchaudio.load(args.input)
        
        # Resampling & Mixdown Mono (necessari per evitare rumore)
        if orig_sr != sample_rate:
            audio = torchaudio.transforms.Resample(orig_sr, sample_rate)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 3. Elaborazione a Pezzi (Chunking)
        audio_tensor = audio.squeeze(0)
        chunk_samples = int(sample_rate * args.chunk_sec)
        
        # Dividiamo l'audio in pezzi (l'ultimo pezzo viene incluso automaticamente)
        chunks = torch.split(audio_tensor, chunk_samples, dim=-1)
        print(f"[*] Durata totale: {audio.shape[1]/sample_rate:.2f}s | Pezzi: {len(chunks)}")

        out_chunks = []

        for i, chunk in enumerate(chunks):
            if chunk.shape[-1] < (sample_rate * 0.1): continue
            
            print(f" > Elaborando pezzo {i+1}/{len(chunks)}...", end="\r")

            # Batch senza il parametro sampling_rate (causa del crash precedente)
            batch = processor(
                audios=[chunk.unsqueeze(0)], 
                descriptions=[args.prompt]
            ).to(device)

            with torch.inference_mode():
                # Autocast gestisce correttamente la precisione bfloat16
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    result = model.separate(
                        batch,
                        predict_spans=False,
                        reranking_candidates=1
                    )
            
            # Riportiamo in float32 per evitare distorsioni nel salvataggio
            out_chunks.append(result.target[0].cpu().float())
            
            del batch, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. Ricostruzione finale
        print(f"\n[*] Unione pezzi e salvataggio finale...")
        final_audio = torch.cat(out_chunks, dim=-1)
        
        if final_audio.ndim == 1:
            final_audio = final_audio.unsqueeze(0)

        # Normalizzazione picco a 0.9 (per rendere l'audio ben udibile)
        peak = final_audio.abs().max()
        if peak > 1e-7:
            final_audio = final_audio / peak * 0.9
            print(f"[+] Picco rilevato: {peak:.4f}. Normalizzazione applicata.")
        else:
            print("⚠️ Il modello non ha trovato corrispondenze nel prompt.")

        torchaudio.save(args.output, final_audio, sample_rate)
        print(f"[✅] COMPLETATO: {args.output}")

    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_separation()