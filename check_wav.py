import sys
import os
import numpy as np
from pathlib import PurePath

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from scipy.io import wavfile
import librosa
import sounddevice as sd


import os
from os import listdir
import pandas as pd
import tensorflow as tf

plt.rcParams["axes.xmargin"] = 0

def get_spectrogram(waveform, type="MEL", NFFT=512, NHOP=64):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=NFFT, frame_step=NHOP)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[:, : NFFT // 2]  # Skip top
    if type == "MEL":
        num_spectrogram_bins = spectrogram.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 10.0, 8000.0, 64
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            16000,
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

    
        spectrogram = mel_spectrograms

    spectrogram = tf.math.log(spectrogram + 1e-7)
    spectrogram = spectrogram / tf.math.reduce_max(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram, ax):
    S = np.squeeze(spectrogram.numpy()).transpose()
    ax.imshow(S, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("Spectrogram")


class Annotate:
    def __init__(self, fname, sr_out=16000):
        self.fname = fname
        self.sr_out = sr_out
        self.y, sr_in = librosa.load(self.fname, mono=True)
        self.y = librosa.resample(self.y, orig_sr=sr_in, target_sr=self.sr_out)

    def process(self):
        def play(event):
            sd.play(self.y, self.sr_out)

        def onPress(event):
            if event.inaxes == self.ax[0]:
                if event.button == event.button.LEFT:
                    print(self.fname)
                    plt.close()
                if event.button == event.button.RIGHT:
                    sd.play(self.y, self.sr_out)

        def next1(event):
            plt.close()

        # Setup figures
        fig, self.ax = plt.subplots(2, 1, figsize=(8,4))
        fig.subplots_adjust(bottom=0.2)
        plt.suptitle(f"{self.fname}")

        ######################
        spectrogram = get_spectrogram(self.y)

        
        # axes[1].set_title('Spectrogram')
        # plt.show()

        # S_fft = np.abs(librosa.stft(self.y, n_fft=self.NFFT, hop_length=self.NHOP))
        # S_mel = librosa.feature.melspectrogram( S=S_fft, sr=self.sr_out, n_fft=self.NFFT, n_mels=self.NMELS, hop_length=self.NHOP, fmin=self.FMIN, fmax=self.FMAX)
        # # S = librosa.stft(self.y, n_fft=self.NFFT, hop_length=self.NHOP)
        # # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        self.ax[0].plot(np.arange(len(self.y)), self.y, linewidth=0.5)
        plot_spectrogram(spectrogram, self.ax[1])
        # self.ax[1].imshow(librosa.amplitude_to_db(S_mel, ref=np.max), origin="lower", aspect="auto", interpolation="nearest")

        # y_harm, y_perc = librosa.effects.hpss(y)

        # Register callbacks for mouse events
        fig.canvas.mpl_connect("button_press_event", onPress)

        ax_save = fig.add_axes([0.8, 0.05, 0.1, 0.05])
        b_save = Button(ax_save, "Next")
        b_save.on_clicked(next1)

        ax_play = fig.add_axes([0.675, 0.05, 0.1, 0.05])
        b_play = Button(ax_play, "Play")
        b_play.on_clicked(play)

        ax_quit = fig.add_axes([0.55, 0.05, 0.1, 0.05])
        b_quit = Button(ax_quit, "Quit")
        b_quit.on_clicked(sys.exit)

        plt.show()


def main():
    data_dirs = "DATA/GUNS"

    files = []
    for fname in os.listdir(data_dirs):
        if fname.endswith(".wav"):
            files.append(fname)
            # print(fname)

    for fname in files:
        ann = Annotate(data_dirs + "/" + fname)
        ann.process()


if __name__ == "__main__":
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
