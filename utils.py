import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa


def get_spectrogram(data, type="MEL", NFFT=512, NHOP=126):
    window = np.hanning(NFFT)

    D = librosa.stft(data, n_fft=NFFT, hop_length=NHOP, window=window)
    MS = librosa.feature.melspectrogram(
        S=np.abs(D),
        sr=16000,
        n_fft=NFFT,
        hop_length=NHOP,
        n_mels=32,#28,
        fmin=10,
        fmax=8000,
    )
    # return librosa.power_to_db(MS, ref=np.max)
    return 20 * np.log10(MS + 0.00001) # Use dBFS


def plot_spectrogram(spectrogram, ax):
    cc = ax.imshow(spectrogram, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("Spectrogram")


def plot_eval(history):
    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        history.epoch,
        100 * np.array(metrics["categorical_accuracy"]),
        100 * np.array(metrics["val_categorical_accuracy"]),
    )
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch")
    plt.ylabel("Loss [CrossEntropy]")
