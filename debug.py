import torch
import torchaudio

def check_file(path):
    try:
        waveform, sr = torchaudio.load(path)
        
        # Statistiche base
        v_max = waveform.abs().max().item()
        v_min = waveform.min().item()
        v_mean = waveform.mean().item()
        v_std = waveform.std().item() # Questa è la vibrazione reale

        print(f"\n--- DIAGNOSI: {path} ---")
        print(f"Sample Rate: {sr} Hz")
        print(f"Picco Massimo (Volume): {v_max:.6f}")
        print(f"Media (DC Offset): {v_mean:.6f}")
        print(f"Deviazione Standard (Vibrazione): {v_std:.6f}")
        
        print("\n[*] Primi 10 campioni del file:")
        print(waveform[0][:10].tolist())

        if v_std < 1e-5 and v_max > 0.1:
            print("\n❌ VERDETTO: DC OFFSET RILEVATO.")
            print("I numeri ci sono ma l'onda è piatta (non vibra).")
        elif v_max < 1e-6:
            print("\n❌ VERDETTO: SILENZIO DIGITALE.")
            print("Il file contiene solo zeri.")
        else:
            print("\n✅ VERDETTO: IL SEGNALE C'È.")
            print("Se non senti nulla, il problema è il formato o la frequenza inudibile.")

    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    import sys
    check_file(sys.argv[1] if len(sys.argv) > 1 else "bonvi_final.wav")