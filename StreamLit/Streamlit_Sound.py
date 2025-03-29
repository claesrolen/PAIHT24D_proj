import numpy as np
import pickle
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams["axes.xmargin"] = 0
import sys
sys.path.append("./..")
from utils import get_spectrogram, plot_spectrogram, plot_eval

@st.cache_data
def getModel():
    print("Loading trained FCN model")
    with open("../GunSound_FCN.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def main_loop():
    ################################################################
    # Load the trained SVM model and scaler
    model = getModel()
    model.pop()

    ################################################################
    # Setup Streamlit widgets
    st.title("Gunshot Detector")
    st.write("")

    fname = st.sidebar.selectbox(
        "Input file",
        (
            "city_2_shot_scene.wav",
            "falling_down_scene.wav",
            "lethal_weapon_scene.wav",
            "miami_street.wav",
            "oneshot_collection_highpass_filter.wav",
            "oneshot_collection.wav",
            "parking_echo.wav",
            "parking.wav",
            "police_arrest.wav",
            "public_area_1.wav",
            "public_area_2.wav",
            "street.wav",
        ),
    )
    
    sr_in0 = librosa.get_samplerate(fname)
    data, sr_in = librosa.load(fname, sr=sr_in0, mono=True)
    data = librosa.resample(data, orig_sr=sr_in, target_sr=16000)
    st.audio(data, sample_rate=16000)

    spectrogram = get_spectrogram(data)

    Q = np.expand_dims(spectrogram, axis=0)
    Q = np.expand_dims(Q, axis=-1)
    y_pred = model.predict(Q)
    yp = np.squeeze(y_pred)

    fig, axes = plt.subplots(3, figsize=(15, 13))
    timescale = np.arange(data.shape[0])
    axes[0].plot(timescale, data)
    axes[0].set_title("Waveform")

    plot_spectrogram(spectrogram, axes[1])

    axes[2].plot(yp[:, 0], label="Gunshot", marker="o")
    axes[2].legend()
    axes[2].set_title("Probability")
    plt.suptitle(f"{fname}")
    st.pyplot(fig)


if __name__ == "__main__":
    main_loop()
