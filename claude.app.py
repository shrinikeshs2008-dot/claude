import io

import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile

# =====================================================================
# LAYER 1: BIOMETRIC SIGNAL PROCESSING ENGINE
# =====================================================================
class VoiceBiometricEngine:
    def __init__(self, sampling_rate: int = 22050):
        self.sr = sampling_rate

    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        """
        Parses a WAV file, extracts Jitter/Shimmer and fundamental pitch (F0)
        to screen for vocal changes linked to thyroid-related fluid accumulation.
        (Demo estimate only - not clinically valid.)
        """
        # --- Properly parse the WAV file ---
        try:
            sr, data = wavfile.read(io.BytesIO(audio_bytes))
            self.sr = int(sr)
            if data.ndim > 1:  # stereo -> mono
                data = data.mean(axis=1)
            waveform = data.astype(np.float32)
            waveform = waveform[: self.sr * 2]  # first 2 seconds only (faster)
            waveform = waveform - np.mean(waveform)
        except Exception:
            waveform = np.array([], dtype=np.float32)

        # Fallback values if audio is empty or corrupted
        if len(waveform) < 100 or np.max(np.abs(waveform)) < 1e-4:
            return {"jitter": 0.001, "shimmer": 0.010, "instability_score": 5.0,
                    "fundamental_pitch_hz": 120.0, "husky_factor": 0.15}

        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)

        # Autocorrelation to find fundamental pitch (F0)
        corr = signal.correlate(waveform, waveform, mode="full")
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-9)

        min_peak_dist = int(self.sr / 300)
        max_peak_dist = int(self.sr / 80)

        peaks, _ = signal.find_peaks(corr[min_peak_dist:max_peak_dist], prominence=0.01)
        peaks = peaks + min_peak_dist

        if len(peaks) < 3:
            return {"jitter": 0.003, "shimmer": 0.014, "instability_score": 10.5,
                    "fundamental_pitch_hz": 95.0, "husky_factor": 0.65}

        periods = np.diff(peaks) / self.sr
        mean_period = np.mean(periods)

        fundamental_pitch = 1.0 / mean_period if mean_period > 0 else 120.0
        jitter = float(np.mean(np.abs(np.diff(periods))) / mean_period)

        valid_peaks = peaks[peaks < len(waveform)]
        if len(valid_peaks) < 3:
            return {"jitter": jitter, "shimmer": 0.012, "instability_score": 15.0,
                    "fundamental_pitch_hz": fundamental_pitch, "husky_factor": 0.40}

        peak_amplitudes = np.abs(waveform[valid_peaks])
        shimmer = float(np.mean(np.abs(np.diff(peak_amplitudes))) / (np.mean(peak_amplitudes) + 1e-9))

        instability_score = min(float((jitter * 750) + (shimmer * 150)), 100.0)
        husky_factor = min(float((shimmer * 4.0) + (jitter * 10.0)), 1.0)

        return {
            "jitter": jitter,
            "shimmer": shimmer,
            "instability_score": round(instability_score, 2),
            "fundamental_pitch_hz": round(float(fundamental_pitch), 1),
            "husky_factor": round(husky_factor, 2),
        }


# =====================================================================
# LAYER 2: BASELINE GENERATOR FOR NEW PATIENTS
# =====================================================================
def create_initial_patient_history(seed_id: int):
    """Generates a default 30-day baseline dataframe for a new profile."""
    np.random.seed(seed_id)
    days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    heart_rate = np.random.normal(74, 3, days)
    spo2 = np.random.normal(98.0, 0.4, days)
    bp_systolic = np.random.normal(122, 4, days)
    voice_instability = np.random.normal(12.2, 1.5, days)
    pitch_hz = np.random.normal(135.0, 5.0, days)

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": heart_rate.round(1),
        "Oxygen_Sat_Percentage": np.clip(spo2, 85, 100).round(1),
        "Systolic_BP_mmHg": bp_systolic.round(1),
        "Voice_Instability_Index": np.clip(voice_instability, 0, 100).round(2),
        "Fundamental_Pitch_Hz": pitch_hz.round(1),
    }).set_index("Date")


# =====================================================================
# LAYER 3: APP UI & MULTI-PATIENT STATE
# =====================================================================
st.set_page_config(page_title="RPMA Dashboard Panel", page_icon="🏥", layout="wide")

st.title("🏥 Remote Patient Monitoring Agent (RPMA)")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Systemic Endocrine & Neuro-Respiratory Tracking Framework.")
st.markdown("---")

if "master_registry" not in st.session_state:
    st.session_state.master_registry = {
        "PT-01": {"name": "Sarah Jenkins", "age": 52, "condition": "Suspected Primary Hypothyroidism",
                  "history": create_initial_patient_history(42)},
        "PT-02": {"name": "David Miller", "age": 29, "condition": "Post-Viral Respiratory Evaluation",
                  "history": create_initial_patient_history(101)},
    }

# --- SIDEBAR: REGISTRATION & NAVIGATION ---
st.sidebar.header("📝 Register New Patient File")
with st.sidebar.form("registration_form", clear_on_submit=True):
    new_name = st.text_input("Full Patient Name:")
    new_age = st.number_input("Age (Years):", min_value=1, max_value=120, value=35)
    new_condition = st.selectbox("Suspected Target Diagnostics Condition:", [
        "Thyroid Fluid Accumulation (Hypothyroidism)",
        "Neuro-Vocal Degradation",
        "General Respiratory Screening",
    ])
    submit_registration = st.form_submit_button("Register Patient")

    if submit_registration and new_name:
        new_id = f"PT-{len(st.session_state.master_registry) + 1:02d}"
        st.session_state.master_registry[new_id] = {
            "name": new_name,
            "age": int(new_age),
            "condition": new_condition,
            "history": create_initial_patient_history(np.random.randint(1, 500)),
        }
        st.sidebar.success(f"Registered {new_name} as ID: {new_id}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Electronic Health Records Selector")
patient_options = {pid: f"{pid} - {meta['name']}" for pid, meta in st.session_state.master_registry.items()}
selected_pid = st.sidebar.selectbox(
    "Active Patient Profile Selection:",
    options=list(patient_options.keys()),
    format_func=lambda x: patient_options[x],
)

# Load active patient
active_patient = st.session_state.master_registry[selected_pid]
clinical_history = active_patient["history"]
latest_vitals = clinical_history.iloc[-1].copy()

# --- SECTION 1: PROFILE HEADER ---
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.markdown(f"👤 **Patient Name:**\n### {active_patient['name']}")
with col_meta2:
    st.markdown(f"🎂 **Chronological Age:**\n### {active_patient['age']} Years")
with col_meta3:
    st.markdown(f"🩺 **Monitored Condition Path:**\n`{active_patient['condition']}`")
with col_meta4:
    st.markdown(f"🆔 **Internal Record Key:**\n`{selected_pid}`")

st.markdown("---")

# --- SECTION 2: INPUTS ---
st.subheader("📥 Log New Vital Diagnostics Entry & Audio Sample")
form_col1, form_col2 = st.columns(2)

with form_col1:
    st.markdown("##### 📝 Input Vital Measurements")
    input_hr = st.number_input("Heart Rate (BPM):", min_value=40, max_value=200,
                               value=int(latest_vitals.get("Heart_Rate_BPM", 72)), step=1, key="hr_in")
    input_bp = st.number_input("Systolic Blood Pressure (mmHg):", min_value=70, max_value=220,
                               value=int(latest_vitals.get("Systolic_BP_mmHg", 120)), step=1, key="bp_in")
    input_spo2 = st.slider("Oxygen Saturation Level - SpO2 (%):", min_value=80.0, max_value=100.0,
                           value=float(latest_vitals.get("Oxygen_Sat_Percentage", 98.0)), step=0.1, key="spo2_in")

with form_col2:
    st.markdown("##### 🔊 Hormonal Acoustic Analysis (Voice Record)")
    uploaded_file = st.file_uploader(
        "Drop patient sustained vowel sound file (.wav) to evaluate Vocal Cord Edema patterns",
        type=["wav"],
    )
    if uploaded_file is not None:
        st.audio(uploaded_file.getvalue(), format="audio/wav")

if st.button("🚀 Commit Log Entry directly to Python Pipeline", use_container_width=True):
    vocal_instability_score = 14.20
    detected_pitch = 130.0
    husky_factor = 0.20

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        engine = VoiceBiometricEngine()
        audio_results = engine.analyze_uploaded_file(file_bytes)
        vocal_instability_score = audio_results["instability_score"]
        detected_pitch = audio_results["fundamental_pitch_hz"]
        husky_factor = audio_results["husky_factor"]
    elif "Thyroid" in active_patient["condition"] or "Hypothyroidism" in active_patient["condition"]:
        # Mock anomalous profile for demonstration when no file is uploaded
        vocal_instability_score = 26.40
        detected_pitch = 88.5
        husky_factor = 0.72

    new_timestamp = pd.Timestamp.now()
    new_entry_row = pd.DataFrame([{
        "Heart_Rate_BPM": float(input_hr),
        "Oxygen_Sat_Percentage": float(input_spo2),
        "Systolic_BP_mmHg": float(input_bp),
        "Voice_Instability_Index": float(vocal_instability_score),
        "Fundamental_Pitch_Hz": float(detected_pitch),
    }], index=[new_timestamp])

    st.session_state.master_registry[selected_pid]["history"] = pd.concat([clinical_history, new_entry_row])

    # Save result so it survives st.rerun()
    st.session_state["last_result"] = {
        "patient": active_patient["name"],
        "pitch": detected_pitch,
        "instability": vocal_instability_score,
        "husky": husky_factor,
    }

    st.toast(f"Data packet merged into record stack for {active_patient['name']}", icon="💾")
    st.rerun()

st.markdown("---")

# --- SECTION 3: RESULTS ---
st.subheader("📊 Analysis Results")

history = st.session_state.master_registry[selected_pid]["history"]

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.caption(f"Latest analysis for: {r['patient']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fundamental Pitch (Hz)", r["pitch"])
    c2.metric("Voice Instability Index", r["instability"])
    c3.metric("Husky Factor", r["husky"])

    if r["pitch"] < 100 and r["husky"] > 0.5:
        st.warning("⚠️ Low pitch with high huskiness – pattern consistent with vocal cord edema. Consider clinical follow-up.")
    else:
        st.success("✅ Voice metrics within the expected range.")
else:
    st.info("Upload a .wav file and click **Commit Log Entry** to see results.")

st.markdown("##### Trends (last 30+ days)")
tab1, tab2, tab3 = st.tabs(["Vitals", "Voice", "Raw data"])
with tab1:
    st.line_chart(history[["Heart_Rate_BPM", "Systolic_BP_mmHg", "Oxygen_Sat_Percentage"]])
with tab2:
    st.line_chart(history[["Voice_Instability_Index", "Fundamental_Pitch_Hz"]])
with tab3:
    st.dataframe(history.tail(15), use_container_width=True)

st.caption("Demo only – not a medical diagnostic tool.")
