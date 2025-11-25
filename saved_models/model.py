# ============================================================
#  SINGLE AUDIO FILE PREDICTION WITH NOISE POLLUTION INDEX
# ============================================================

import os
os.environ['TFHUB_CACHE_DIR'] = './tfhub_cache'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pickle
import warnings

warnings.filterwarnings('ignore')


A_WEIGHTING_CLASS = {
    "traffic": -1.3,        # low-mid frequency rumble
    "construction": -2.0,   # loud mid-high noise
    "ambient": -3.0         # mixed, mostly soft background
}

NOISE_CLASS_WEIGHT = {
    "traffic": 1.3,         # harmful, common noise
    "construction": 1.5,    # very harmful, sharp
    "ambient": 0.7          # mostly harmless
}

# APPROXIMATE A-WEIGHTING TECHNIQUE
def apply_approx_a_weight(db_level: float, noise_class: str) -> float:
    key = noise_class.lower()
    corr = A_WEIGHTING_CLASS.get(key, 0.0)
    return db_level + corr

# Combine A-weighted result with class annoyance/harmfulness weight
def apply_noise_class_weight(a_weighted_db: float, noise_class: str) -> float:
    key = noise_class.lower()
    w = NOISE_CLASS_WEIGHT.get(key, 1.0)
    return a_weighted_db * w

# Computation of the Noise Pollution Index
def compute_npi(weighted_db: float) -> float:
    min_db, max_db = 30.0, 90.0
    x = max(min_db, min(max_db, weighted_db))
    npi = 1.0 + 9.0 * (x - min_db) / (max_db - min_db)
    return round(npi, 2)


# ============================================================
#  Load Model & Dependencies
# ============================================================
print("Loading model...")

# Get the directory where this script is located
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained classifier (safer: compile=False to avoid deserialization/compile issues)
model_path = os.path.join(_script_dir, 'merged_best_tf211_fixed.h5')
print("Loading classifier model from:", model_path)
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully (compile=False).")
except Exception as e:
    print("Failed to load model with compile=False, trying default load... Error:", e)
    model = tf.keras.models.load_model(model_path)
    print("Model loaded (fallback).")

# Load label encoder
label_path = os.path.join(_script_dir, 'label_encoder.pkl')
with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)
print(f"Label encoder loaded - Classes: {label_encoder.classes_}")

# Load YAMNet
print("Loading YAMNet feature extractor...")
yamnet_model = hub.load('./saved_models/yamnet')
print("YAMNet loaded.\n")


# ============================================================
#  Audio Loading Function
# ============================================================
def load_wav_16k_mono(filename):
    """Load audio file and resample to 16kHz mono."""
    audio, sr = librosa.load(filename, sr=16000, mono=True)
    return tf.constant(audio, dtype=tf.float32), audio, sr


# ============================================================
#  Decibel Level Calculation (RMS-based SPL Estimation)
# ============================================================
def calculate_db_level(audio, sr=16000):
    """
    Calculate decibel level (dB SPL - Sound Pressure Level) using RMS method.
    This provides a more realistic estimate of actual sound pressure levels.

    For normalized audio (librosa loads as [-1, 1]), we estimate SPL by:
    1. Calculating RMS of the audio signal
    2. Converting to dB relative to a reference level
    3. Scaling to realistic SPL range (approximately 30-120 dB)

    Typical mappings:
    - RMS ≈ 0.01 → ~50-60 dB (quiet room)
    - RMS ≈ 0.1 → ~70-80 dB (normal conversation, moderate traffic)
    - RMS ≈ 0.3 → ~85-95 dB (loud traffic, construction)
    - RMS ≈ 0.5+ → ~95-110 dB (very loud, hazardous)
    """
    # Step 1: Calculate RMS (Root Mean Square) of the audio signal
    rms = np.sqrt(np.mean(audio ** 2))

    # Step 2: Avoid log(0)
    eps = 1e-10
    rms = max(rms, eps)

    # Step 3: Convert RMS to dB SPL
    # For normalized audio, we use a conversion that maps RMS to realistic SPL
    # Formula: dB = offset + (20 * log10(RMS))
    # We calibrate so that:
    #   - RMS = 0.01 → ~50 dB (quiet)
    #   - RMS = 0.1 → ~75 dB (moderate)
    #   - RMS = 0.5 → ~95 dB (loud)

    # Calculate dB from RMS
    # Using: dB = 94 + 20*log10(RMS) where 94 dB is full-scale reference
    # This maps RMS=1.0 → 94 dB, RMS=0.1 → 74 dB, RMS=0.01 → 54 dB
    db_level = 94 + (20 * np.log10(rms))

    # However, for quiet recordings, this might still be too low
    # So we add a base offset to better represent real-world scenarios
    # Adjust based on typical phone/computer microphone sensitivity
    db_level = db_level + 10  # Add 10 dB offset for typical recording conditions

    # Step 4: Clamp to realistic practical limits (35-115 dB SPL)
    # Real-world SPL ranges:
    # - Quiet room: 30-40 dB
    # - Normal conversation: 60-70 dB
    # - Traffic: 70-85 dB
    # - Construction: 85-100 dB
    # - Very loud: 100-115 dB
    db_level = max(35, min(115, db_level))

    return db_level


# ============================================================
#  Noise Pollution Category Function
# ============================================================
def get_pollution_category(db_level):
    """Categorize noise level into Safe, Acceptable, or Hazardous."""
    if db_level <= 70:
        return {
            'category': 'Safe',
            'health_impact': 'No risk - Safe for prolonged exposure',
            'recommendation': 'Normal noise level, no action needed'
        }
    elif 71 <= db_level <= 90:
        return {
            'category': 'Acceptable',
            'health_impact': 'Moderate risk - May cause discomfort',
            'recommendation': 'Limit exposure time, use hearing protection if prolonged'
        }
    else:  # db_level >= 91
        return {
            'category': 'Hazardous',
            'health_impact': 'High risk - Can cause immediate hearing damage',
            'recommendation': 'Evacuate area or use hearing protection immediately'
        }


# ============================================================
#  Prediction Function (Single Audio File)
# ============================================================
def predict_audio(audio_path):
    """Predict noise class and pollution index for a single audio file."""
    print("=" * 60)
    print("AUDIO ANALYSIS & PREDICTION")
    print("=" * 60)

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return None

    print(f"File: {os.path.basename(audio_path)}")
    print(f"Path: {audio_path}")

    try:
        # Load and preprocess audio (librosa.load already resamples to 16k and mono due to our helper)
        print("\nProcessing audio...")
        wav_tensor, audio_array, sr = load_wav_16k_mono(audio_path)

        # Safety checks: ensure dtype and shape
        if hasattr(wav_tensor, "numpy"):
            waveform = wav_tensor.numpy()
        else:
            waveform = np.array(wav_tensor, dtype=np.float32)

        # Ensure 1-D mono
        if waveform.ndim == 2:
            print("Stereo detected. Converting to mono by averaging channels.")
            waveform = np.mean(waveform, axis=1)

        # Ensure float32 and normalized to [-1, 1]
        waveform = waveform.astype(np.float32)
        max_val = np.max(np.abs(waveform)) if waveform.size else 0.0
        if max_val > 0:
            waveform = waveform / (max_val + 1e-9)

        # Ensure sampling rate is 16k (load_wav_16k_mono should handle this already)
        if sr != 16000:
            print(f"Warning: sample rate is {sr}, expected 16000. Resampling.")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Calculate dB level
        db_level = calculate_db_level(audio_array, sr)
        pollution_info = get_pollution_category(db_level)

        # YAMNet expects a 1-D float32 waveform at 16 kHz.
        # Some TF Hub YAMNet wrappers accept either numpy array or tensor.
        # Pass numpy array explicitly for consistency.
        print("Calling YAMNet (waveform length):", waveform.shape)
        try:
            scores, embeddings, spectrogram = yamnet_model(waveform)
        except Exception as ye:
            # try passing as tensor if direct numpy fails
            print("YAMNet call with numpy failed, trying tensor input. Error:", ye)
            scores, embeddings, spectrogram = yamnet_model(tf.constant(waveform, dtype=tf.float32))

        # embeddings shape: (num_frames, 1024) — we take mean over frames
        emb = tf.reduce_mean(embeddings, axis=0)  # shape (1024,)
        emb = tf.expand_dims(emb, 0)  # shape (1, 1024)
        print("Embedding tensor shape:", emb.shape)

        # Convert embedding to numpy before passing to Keras model
        if hasattr(emb, "numpy"):
            emb_np = emb.numpy()
        else:
            emb_np = np.array(emb)

        print("Embedding numpy shape:", emb_np.shape, "dtype:", emb_np.dtype)

        # Predict noise type - ensure shapes/dtypes are compatible
        predictions = model.predict(emb_np, verbose=0)
        print("Classifier predictions shape:", np.shape(predictions))

        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

        # Safer label decoding
        if hasattr(label_encoder, "classes_"):
            predicted_class = label_encoder.classes_[predicted_idx]
        else:
            # fallback to inverse_transform if label encoder is different
            predicted_class = label_encoder.inverse_transform([predicted_idx])[0]

        # Collect probabilities
        all_probs = {}
        if hasattr(label_encoder, "classes_"):
            for i in range(len(label_encoder.classes_)):
                all_probs[label_encoder.classes_[i]] = float(predictions[0][i])
        else:
            # generic fallback
            for i, p in enumerate(predictions[0]):
                all_probs[str(i)] = float(p)

        a_weighted_db = apply_approx_a_weight(db_level, predicted_class)
        weighted_db = apply_noise_class_weight(a_weighted_db, predicted_class)
        npi_score = compute_npi(weighted_db)

        # Print results (same as before)
        print("\n" + "=" * 60)
        print("NOISE CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"\nNoise Type: {predicted_class.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")

        if confidence > 0.8:
            print("Prediction Reliability: HIGH")
        elif confidence > 0.5:
            print("Prediction Reliability: MEDIUM")
        else:
            print("Prediction Reliability: LOW")

        print("\n" + "=" * 60)
        print("NOISE POLLUTION INDEX")
        print("=" * 60)
        print(f"\nDecibel Level: {db_level:.1f} dB")
        print(f"A-weighted (approx): {a_weighted_db:.1f} dB")
        print(f"Weighted Level (A + class): {weighted_db:.1f}")
        print(f"NPI Score (1 - 10): {npi_score:.2f}")
        print(f"Status: {pollution_info['category'].upper()}")
        print(f"Health Impact: {pollution_info['health_impact']}")
        print(f"Recommendation: {pollution_info['recommendation']}")

        return {
            'file': os.path.basename(audio_path),
            'noise_type': predicted_class,
            'confidence': float(confidence),
            'db_level': float(db_level),
            'a_weighted_db': float(a_weighted_db),
            'weighted_db': float(weighted_db),
            'npi_score': float(npi_score),
            'pollution_category': pollution_info['category'],
            'health_impact': pollution_info['health_impact'],
            'recommendation': pollution_info['recommendation'],
            'all_probabilities': all_probs
        }

    except Exception as e:
        print("Error in predict_audio():", str(e))
        import traceback
        traceback.print_exc()
        return None

# No top-level execution in library module