import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    # used for V2C-Net from Chenqi, et.al
    if len(data) == 15:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            #
            spks,
            emotions,
            emos,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        #
        spks = torch.from_numpy(spks).float().to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        emos = torch.from_numpy(emos).float().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            #
            spks,
            emotions,
            emos,
        )

    """
    [our method] V2C-Net is coarse-grained, that is, all frames of the entire video are used as an emotion embedding vector, \
        regardless of whether there are redundant speakers in the video (A movie scene often has multiple characters talking or interacting at the same time). \
            We provide the V2C-Animation 2.0 version, which annotates the only speakers in the video, as well as their lip and faces regions. 
    """
    if len(data) == 19:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            #
            spks,
            emotions,
            emos,
            feature_256,
            lip_lens,
            max_lip_lens,
            lip_embedding,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        lip_lens = torch.from_numpy(lip_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        #
        spks = torch.from_numpy(spks).float().to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        emos = torch.from_numpy(emos).float().to(device)
        feature_256 = torch.from_numpy(feature_256).float().to(device)
        lip_embedding = torch.from_numpy(lip_embedding).float().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            #
            spks,
            emotions,
            emos,
            feature_256,
            lip_lens,
            max_lip_lens,
            lip_embedding,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)

    # for synthesize.py
    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, emotions) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        emotions = torch.from_numpy(emotions).long().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, emotions)

    # for synthesize.py
    if len(data) == 12:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, emotions, \
            mels, pitches, energies, durations, mel_lens) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        # 
        mels = torch.from_numpy(mels).float().to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, emotions, \
            mels, pitches, energies, durations, mel_lens)


def log(
    logger, step=None, losses=None, fig=None, audio=None, accs_val_spk=None, \
    accs_train_spk=None, accs_val_emo=None, accs_train_emo=None, \
    avg_mcd_val=None, avg_mcd_train=None, \
    sampling_rate=16000, tag="", LM=None,
):
    if losses is not None:
        if LM == "Chem":
            logger.add_scalar("Loss/total_loss", losses[0], step)
            logger.add_scalar("Loss/mel_loss", losses[1], step)
            logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
            logger.add_scalar("Loss/pitch_loss", losses[3], step)
            logger.add_scalar("Loss/energy_loss", losses[4], step)
            logger.add_scalar("Loss/pitch_MAE", losses[5], step)
            logger.add_scalar("Loss/energy_MAE", losses[6], step)
            logger.add_scalar("Loss/mel_MSE", losses[7], step)
            logger.add_scalar("Loss/mel_postnet_MSE", losses[8], step)
        if LM == "V2C":
            logger.add_scalar("Loss/total_loss", losses[0], step)
            logger.add_scalar("Loss/Mel_mae", losses[1], step)
            logger.add_scalar("Loss/Mel_Post_mae", losses[2], step)
            logger.add_scalar("Loss/pitch_MSE", losses[3], step)
            logger.add_scalar("Loss/energy_MSE", losses[4], step)
            logger.add_scalar("Loss/pitch_MAE", losses[5], step)
            logger.add_scalar("Loss/energy_MAE", losses[6], step)
            logger.add_scalar("Loss/Mel_mse", losses[7], step)
            logger.add_scalar("Loss/Mel_Post_mse", losses[8], step)
            logger.add_scalar("Loss/Emo_Cross_MSE", losses[9], step)
            logger.add_scalar("Loss/CTC_MDA_video", losses[10], step)
            logger.add_scalar("Loss/CTC_MEL", losses[11], step)

    if accs_val_spk is not None:
        logger.add_scalar("Acc/eval_acc_rec_spk", accs_val_spk[0], step)
        logger.add_scalar("Acc/eval_acc_pred_spk", accs_val_spk[1], step)
    if accs_train_spk is not None:
        logger.add_scalar("Acc/train_acc_rec_spk", accs_train_spk[0], step)
        logger.add_scalar("Acc/train_acc_pred_spk", accs_train_spk[1], step)
    if accs_val_emo is not None:
        logger.add_scalar("Acc/eval_acc_rec_emo", accs_val_emo[0], step)
        logger.add_scalar("Acc/eval_acc_pred_emo", accs_val_emo[1], step)
    if accs_train_emo is not None:
        logger.add_scalar("Acc/train_acc_rec_emo", accs_train_emo[0], step)
        logger.add_scalar("Acc/train_acc_pred_emo", accs_train_emo[1], step)
    if avg_mcd_val is not None:
        logger.add_scalar("MCD/val_avg_mcd", avg_mcd_val[0], step)
        logger.add_scalar("MCD/val_avg_mcd_dtw", avg_mcd_val[1], step)
        logger.add_scalar("MCD/val_avg_mcd_dtw_sl", avg_mcd_val[2], step)
    if avg_mcd_train is not None:
        logger.add_scalar("MCD/train_avg_mcd", avg_mcd_train[0], step)
        logger.add_scalar("MCD/train_avg_mcd_dtw", avg_mcd_train[1], step)
        logger.add_scalar("MCD/train_avg_mcd_dtw_sl", avg_mcd_train[2], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def generate_square_subsequent_mask(sz1,sz2):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz1, sz2), diagonal=2) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    basename = targets[0][0]
    src_len = predictions[6][0].item()
    mel_len = predictions[7][0].item()
    mel_len_gt = targets[7][0].item()

    mel_target = targets[6][0, :mel_len_gt].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)

    pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


# write to calcuate the acc
def synth_multi_samples(targets, predictions, vocoder, model_config, preprocess_config):
    
    basenames = targets[0]
    speakers = targets[2]
    emotions = targets[13]
    # 
    wav_reconstructions = []
    wav_predictions = []
    cofs = []  # for mcd-dtw-sl
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[6][i].item()
        mel_len = predictions[7][i].item()
        mel_len_gt = targets[7][i].item()
        mel_target = targets[6][i, :mel_len_gt].detach().transpose(0, 1)
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration_sum = targets[11][i].detach().cpu().numpy().item()
        duration_prediction_sum = targets[16][i].item()*4
        cofs.append((duration_sum, duration_prediction_sum))

        if vocoder is not None:
            from .model import vocoder_infer

            wav_reconstruction = vocoder_infer(
                mel_target.unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
            wav_prediction = vocoder_infer(
                mel_prediction.unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
        else:
            wav_reconstruction = wav_prediction = None

        wav_reconstructions.append(wav_reconstruction)
        wav_predictions.append(wav_prediction)

    return wav_reconstructions, wav_predictions, basenames, speakers, emotions, cofs


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
