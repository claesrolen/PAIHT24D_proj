import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa


def get_spectrogram(data, type="MEL", NFFT=512, NHOP=126):
    window = np.hanning(NFFT)

    D = librosa.stft(data, n_fft=NFFT, hop_length=NHOP, window=window)
    D = 2 * (np.abs(D)**2) / np.sum(window)
    MS = librosa.feature.melspectrogram(
        S=D,
        sr=16000,
        n_fft=NFFT,
        hop_length=NHOP,
        n_mels=28,
        fmin=10,
        fmax=8000,
    )
    
    # MS = librosa.feature.melspectrogram(
    #     y=data,
    #     sr=16000,
    #     n_fft=NFFT,
    #     hop_length=NHOP,
    #     n_mels=24,
    #     fmin=10,
    #     fmax=8000,
    # )
    # return librosa.power_to_db(MS, ref=np.max)
    # MS = librosa.feature.melspectrogram(
    #     y=data,
    #     sr=16000,
    #     n_fft=NFFT,
    #     hop_length=150,
    #     n_mels=28,
    #     fmin=10,
    #     fmax=8000,
    # )
    return 10*np.log10(MS+0.00001)


# def get_spectrogram(waveform, type="MEL", NFFT=512, NHOP=105):
#     # Convert the waveform to a spectrogram via a STFT.
#     spectrogram = tf.signal.stft(waveform, frame_length=NFFT, frame_step=NHOP)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = spectrogram[:, : NFFT // 2]  # Skip top
#     if type == "MEL":
#         num_spectrogram_bins = spectrogram.shape[-1]
#         lower_edge_hertz, upper_edge_hertz, num_mel_bins = 10.0, 8000.0, 24
#         linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#             num_mel_bins,
#             num_spectrogram_bins,
#             16000,
#             lower_edge_hertz,
#             upper_edge_hertz,
#         )
#         mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
#         mel_spectrograms.set_shape(
#             spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
#         )

#         # SR = 16000
#         # NMEL = 24
#         # FMIN = 10
#         # FMAX = 8000
#         # mel_weights = tf.signal.linear_to_mel_weight_matrix(
#         #     NMEL, NFFT // 2, SR, FMIN, FMAX
#         # )
#         # mel_spectrogram = tf.tensordot(spectrogram, mel_weights, 1)
#         # mel_spectrogram.set_shape([spectrogram.shape[0], NMEL])
#         spectrogram = mel_spectrograms

#     spectrogram = tf.math.log(spectrogram + 1e-7)
#     spectrogram = spectrogram / tf.math.reduce_max(spectrogram)
#     spectrogram = spectrogram[..., tf.newaxis]
#     return spectrogram


def plot_spectrogram(spectrogram, ax):
    # S = np.squeeze(spectrogram.numpy()).transpose()
    # spectrogram = 10*np.log10(spectrogram) #librosa.power_to_db(spectrogram, ref=np.max)
    cc = ax.imshow(spectrogram, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("Spectrogram")
    # plt.colorbar(cc)


def plot_eval(history):
    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        history.epoch,
        100 * np.array(metrics["accuracy"]),
        100 * np.array(metrics["val_accuracy"]),
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
