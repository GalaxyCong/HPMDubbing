import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import torch
import random
import json
import tgt
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import audio as Audio
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
# import sys
# import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0

# from meldataset import mel_spectrogram_hifigan

with open(
        os.path.join("/data/conggaoxiang/V2C/V2C_Code/public_processed_data_Origin/MovieAnimation", "val.txt"), "r", encoding="utf-8"
) as f:
    valname = []
    for line in f.readlines():
        n, s, t, r = line.strip("\n").split("|")
        valname.append(n)


with open(
        os.path.join("/data/conggaoxiang/V2C/V2C_Code/public_processed_data_Origin/MovieAnimation", "train.txt"), "r", encoding="utf-8"
) as f:
    trainname = []
    for line in f.readlines():
        n, s, t, r = line.strip("\n").split("|")
        trainname.append(n)


# with open(
#         os.path.join("/data/conggaoxiang/chemistry_lectures/chem_release_version/data_splits", "val.txt"), "r", encoding="utf-8"
# ) as f:
#     valname = []
#     for line in f.readlines():
#         n, s = line.strip("\n").split("|")
#         valname.append(n)
#
#
# with open(
#         os.path.join("/data/conggaoxiang/chemistry_lectures/chem_release_version/data_splits", "train.txt"), "r", encoding="utf-8"
# ) as f:
#     trainname = []
#     for line in f.readlines():
#         n, s = line.strip("\n").split("|")
#         trainname.append(n)
#
#
# with open(
#         os.path.join("/data/conggaoxiang/chemistry_lectures/chem_release_version/data_splits", "test.txt"), "r", encoding="utf-8"
# ) as f:
#     testname = []
#     for line in f.readlines():
#         n, s = line.strip("\n").split("|")
#         testname.append(n)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        print("self.out_dir:", self.out_dir)
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        # global pitch_R, energy_R, n
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        my_num = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                """V2C and Chem dataset"""
                # """V2C"""
                if basename in trainname or basename in valname:
                # """Chem"""
                # if basename in trainname or basename in valname or basename in testname:
                    my_num = my_num+1
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    # print(tg_path)
                    # assert False
                    if os.path.exists(tg_path):
                        ret = self.process_utterance(speaker, basename)
                        if ret is None:
                            print("if ret is None:", tg_path)
                            continue
                        else:
                            info, pitch_R, energy_R, n = ret
                        out.append(info)

                        if len(pitch_R) > 0:
                            pitch_scaler.partial_fit(pitch_R.reshape((-1, 1)))
                        if len(energy_R) > 0:
                            energy_scaler.partial_fit(energy_R.reshape((-1, 1)))
                        n_frames += n

                    else:
                        print("tg_path:", tg_path)
                # """GRID"""
                # my_num = my_num + 1
                # tg_path = os.path.join(
                #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                # )
                # # print(tg_path)
                # # assert False
                # if os.path.exists(tg_path):
                #     ret = self.process_utterance(speaker, basename)
                #     if ret is None:
                #         print("if ret is None:", tg_path)
                #         continue
                #     else:
                #         info, pitch_R, energy_R, n = ret
                #     out.append(info)
                #
                #     if len(pitch_R) > 0:
                #         pitch_scaler.partial_fit(pitch_R.reshape((-1, 1)))
                #     if len(energy_R) > 0:
                #         energy_scaler.partial_fit(energy_R.reshape((-1, 1)))
                #     n_frames += n


        print("Computing statistic quantities ...")
        print("my_num:", my_num)
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train1.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val1.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.txt".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        # wav, _ = librosa.load(wav_path, sr=16000)
        sampling_rate_, wav = read(wav_path)
        wav = wav / MAX_WAV_VALUE

        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)
        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None
        _, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        wav = torch.FloatTensor(wav).to(device)
        wav = wav.unsqueeze(0)
        # Compute mel-scale spectrogram and energy
        mel_spectrogram, _ = mel_spectrogram_HF(y=wav, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                                                win_size=1024, fmin=0, fmax=8000)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        _, mel_length = mel_spectrogram.shape
        if mel_length < sum(duration):
            mel_spectrogram = mel_spectrogram[:, : mel_length]
            energy = energy[: mel_length]
            pitch = pitch[: mel_length]
            duration[-1] = duration[-1] - (sum(duration)-mel_length)
        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        # sil_phones = []

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value



mel_basis = {}
hann_window = {}
def mel_spectrogram_HF(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    magnitudes = spec

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    energy = torch.norm(magnitudes, dim=1)

    melspec = torch.squeeze(spec, 0).cpu().numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).cpu().numpy().astype(np.float32)

    return melspec, energy

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

