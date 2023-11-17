"""
Function: If you have enough GPU memory, this will help you speed up inference
Date: 2023/11/12
Author: HPMDubbing
"""

import argparse
import os
from resemblyzer import VoiceEncoder
import torch
import yaml
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_multi_samples
from dataset import Dataset

import numpy as np

from scipy.io.wavfile import write
from tqdm import tqdm
import sys
from mcd import Calculate_MCD

sys.path.append("..")
from resemblyzer import preprocess_wav
from joblib import Parallel, delayed

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acc_metric(speakers_ids, speakers_all, wav_reconstructions_utterance_embeds, \
               wav_predictions_utterance_embeds, ids2loc_map, loc2ids_map, centroids=None):
    if centroids is None:
        # Inclusive centroids (1 per speaker) (speaker_num x embed_size)
        centroids_rec = np.zeros((len(speakers_ids), wav_reconstructions_utterance_embeds.shape[1]), dtype=np.float)
        # calculate the centroids for each speaker
        counters = np.zeros((len(speakers_ids),))
        for i in range(wav_reconstructions_utterance_embeds.shape[0]):
            # calculate centroids
            centroids_rec[ids2loc_map[speakers_all[i].item()]] += wav_reconstructions_utterance_embeds[i]
            counters[ids2loc_map[speakers_all[i].item()]] += 1
        # normalize
        for i in range(len(counters)):
            centroids_rec[i] = centroids_rec[i] / counters[i]
            centroids_rec[i] = centroids_rec[i] / (np.linalg.norm(centroids_rec[i], ord=2) + 1e-5)
    else:
        centroids_rec = centroids
    sim_matrix_pred = np.dot(wav_predictions_utterance_embeds, centroids_rec.T)
    sim_matrix_rec = np.dot(wav_reconstructions_utterance_embeds, centroids_rec.T)
    # pred_locs 512x1
    pred_locs = sim_matrix_pred.argmax(axis=1)
    rec_locs = sim_matrix_rec.argmax(axis=1)
    # calculate acc
    correct_num_pred = 0
    correct_num_rec = 0
    for i in range(len(pred_locs)):
        if loc2ids_map[pred_locs[i]] == speakers_all[i].item():
            correct_num_pred += 1
        if loc2ids_map[rec_locs[i]] == speakers_all[i].item():
            correct_num_rec += 1
    eval_acc_pred = correct_num_pred / float(len(pred_locs))
    eval_acc_rec = correct_num_rec / float(len(pred_locs))

    return eval_acc_rec, eval_acc_pred


def assess_all_spk_emo(encoder_spk, encoder_emo, sampling_rate, samples_path,
                       mcd_box_plain, mcd_box_dtw, mcd_box_adv_dtw,
                       wav_reconstructions_batch, wav_predictions_batch, tags_batch, speakers_batch, emotions_batch,
                       cofs_batch):
    # how many speaker in here (value equal to the speaker id)
    speakers_ids = torch.unique(torch.tensor(speakers_batch, dtype=torch.long))
    emotions_ids = torch.unique(torch.tensor(emotions_batch, dtype=torch.long))

    # speakers mapping
    ids2loc_map = {}
    loc2ids_map = {}
    for i in range(len(speakers_ids)):
        ids2loc_map[speakers_ids[i].item()] = i
        loc2ids_map[i] = speakers_ids[i].item()
    # emotion mapping
    ids2loc_map_emo = {}
    loc2ids_map_emo = {}
    for i in range(len(emotions_ids)):
        ids2loc_map_emo[emotions_ids[i].item()] = i
        loc2ids_map_emo[i] = emotions_ids[i].item()

    # save and reload val (train) samples
    # save
    rec_fpaths = []
    pred_fpaths = []
    for i in range(len(wav_reconstructions_batch)):
        rec_fpath = os.path.join(samples_path, "wav_rec_{}.wav".format(tags_batch[i]))
        pred_fpath = os.path.join(samples_path, "wav_pred_{}.wav".format(tags_batch[i]))

        write(rec_fpath, sampling_rate, wav_reconstructions_batch[i])
        write(pred_fpath, sampling_rate, wav_predictions_batch[i])

        rec_fpaths.append(rec_fpath)
        pred_fpaths.append(pred_fpath)
    # print("Reloading ...")
    rec_wavs = np.array(
        list(map(preprocess_wav, tqdm(rec_fpaths, "Preprocessing rec wavs", len(rec_fpaths), disable=True))),
        dtype=object)
    pred_wavs = np.array(
        list(map(preprocess_wav, tqdm(pred_fpaths, "Preprocessing pred wavs", len(pred_fpaths), disable=True))),
        dtype=object)
    for i in tqdm(range(len(rec_fpaths)), disable=True):
        if i != (len(rec_fpaths) - 1):
            mcd_box_plain.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=False)
            mcd_box_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=False)
            mcd_box_adv_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), cofs_batch[i], average=False)
        else:
            avg_mcd_plain = mcd_box_plain.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=True)
            avg_mcd_dtw = mcd_box_dtw.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=True)
            avg_mcd_adv_dtw = mcd_box_adv_dtw.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), cofs_batch[i], average=True)

    # speaker and emotion: (speakers/emttion_per_batch x utterances_per_se) x embedding_dim
    # Compute the wav embedding for accuracy (spk)
    wav_reconstructions_utterance_embeds_spk = np.array(list(map(encoder_spk.embed_utterance, rec_wavs)))
    wav_predictions_utterance_embeds_spk = np.array(list(map(encoder_spk.embed_utterance, pred_wavs)))
    # Compute the wav embedding for accuracy (emo)
    wav_reconstructions_utterance_embeds_emo = np.array(list(map(encoder_emo.embed_utterance, rec_wavs)))
    wav_predictions_utterance_embeds_emo = np.array(list(map(encoder_emo.embed_utterance, pred_wavs)))

    # calcuate accuracy
    # emotion
    eval_acc_rec_emo, eval_acc_pred_emo = acc_metric(emotions_ids, emotions_batch, \
                                                     wav_reconstructions_utterance_embeds_emo,
                                                     wav_predictions_utterance_embeds_emo, \
                                                     ids2loc_map_emo, loc2ids_map_emo, centroids=None)
    # speaker
    eval_acc_rec_spk, eval_acc_pred_spk = acc_metric(speakers_ids, speakers_batch, \
                                                     wav_reconstructions_utterance_embeds_spk,
                                                     wav_predictions_utterance_embeds_spk, \
                                                     ids2loc_map, loc2ids_map)

    return avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw, eval_acc_rec_spk, eval_acc_pred_spk, eval_acc_rec_emo, eval_acc_pred_emo


def Test_more(batchs=None, encoder_spk=None, encoder_emo=None, loader=None, sampling_rate=None, samples_path=None,
              mcd_box_plain=None, mcd_box_dtw=None, mcd_box_adv_dtw=None, useGT=False):
    wav_reconstructions_batch = []
    wav_predictions_batch = []
    tags_batch = []
    speakers_batch = []
    emotions_batch = []
    cofs_batch = []

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*(batch[2:]), useGT=useGT)
            # synthesize multiple sample for speaker and emotion accuracy calculation
            wav_reconstructions, wav_predictions, tags, speakers, emotions, cofs = synth_multi_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config2,
            )
            # merge
            wav_reconstructions_batch.extend(wav_reconstructions)
            wav_predictions_batch.extend(wav_predictions)
            tags_batch.extend(tags)
            speakers_batch.extend(speakers)
            emotions_batch.extend(emotions)
            cofs_batch.extend(cofs)

    # calculate metrics
    os.makedirs(samples_path, exist_ok=True)
    avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw, eval_acc_rec_spk, eval_acc_pred_spk, eval_acc_rec_emo, eval_acc_pred_emo = assess_all_spk_emo(
        encoder_spk=encoder_spk, encoder_emo=encoder_emo, sampling_rate=sampling_rate, samples_path=samples_path,
        mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw,
        wav_reconstructions_batch=wav_reconstructions_batch, wav_predictions_batch=wav_predictions_batch,
        tags_batch=tags_batch, speakers_batch=speakers_batch, emotions_batch=emotions_batch, cofs_batch=cofs_batch)

    return avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw, eval_acc_rec_spk, eval_acc_pred_spk, eval_acc_rec_emo, eval_acc_pred_emo


def calculate_all_acc(preprocess_config2, model_config, model, vocoder, \
                      encoder_spk, encoder_emo, loader, sampling_rate=None, samples_path=None, \
                      mcd_box_plain=None, mcd_box_dtw=None, mcd_box_adv_dtw=None, useGT=False):
    results = Parallel(n_jobs=8, verbose=1)(
        delayed(Test_more)(batchs, encoder_spk, encoder_emo, loader, sampling_rate, samples_path, \
                           mcd_box_plain, mcd_box_dtw, mcd_box_adv_dtw, useGT) for batchs in loader
    )

    avg_mcd_plain = sum(result[0] for result in results)
    avg_mcd_dtw = sum(result[1] for result in results)
    avg_mcd_adv_dtw = sum(result[2] for result in results)
    eval_acc_rec_spk = sum(result[3] for result in results)
    eval_acc_pred_spk = sum(result[4] for result in results)
    eval_acc_rec_emo = sum(result[5] for result in results)
    eval_acc_pred_emo = sum(result[6] for result in results)

    return avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw, eval_acc_rec_spk, eval_acc_pred_spk, eval_acc_rec_emo, eval_acc_pred_emo


def save_wav(sampling_rate, samples_path,
             wav_reconstructions_batch, wav_predictions_batch, tags_batch):
    rec_fpaths = []
    pred_fpaths = []
    for i in range(len(wav_reconstructions_batch)):
        generated_path = os.path.join(samples_path, "generated_path")
        reconstruct_path = os.path.join(samples_path, "reconstruct_path")
        os.makedirs(generated_path, exist_ok=True)
        os.makedirs(reconstruct_path, exist_ok=True)

        rec_fpath = os.path.join(reconstruct_path, "wav_rec_{}.wav".format(tags_batch[i]))
        pred_fpath = os.path.join(generated_path, "wav_pred_{}.wav".format(tags_batch[i]))

        write(rec_fpath, sampling_rate, wav_reconstructions_batch[i])
        write(pred_fpath, sampling_rate, wav_predictions_batch[i])

        rec_fpaths.append(rec_fpath)
        pred_fpaths.append(pred_fpath)


def evaluate_all_valset(model, step, configs, vocoder=None, encoder_spk=None, encoder_emo=None):
    preprocess_config, model_config, train_config, preprocess_config2 = configs
    useGT = False

    val_samples_path = train_config["path"]["result_path"]
    sampling_rate = preprocess_config2["preprocessing"]["audio"]["sampling_rate"]

    dataset_val = Dataset(
        "val.txt", preprocess_config2, train_config, sort=False, drop_last=False, diff_audio=False
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset_val.collate_fn,
    )

    print("Start load all val-set", '\n')
    print('The number of the val-set:', len(dataset_val), '\n')
    print('The number of loader_val:', len(loader_val), '\n')
    print("calculate all acc ...")
    # initialize MCD module
    mcd_box_plain = Calculate_MCD("plain", sr=sampling_rate)
    mcd_box_dtw = Calculate_MCD("dtw", sr=sampling_rate)
    mcd_box_adv_dtw = Calculate_MCD("adv_dtw", sr=sampling_rate)

    avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw, eval_acc_rec_spk, eval_acc_pred_spk, eval_acc_rec_emo, eval_acc_pred_emo = calculate_all_acc(
        preprocess_config2, model_config, model, vocoder, encoder_spk, encoder_emo, loader_val,
        sampling_rate=sampling_rate,
        samples_path=val_samples_path, \
        mcd_box_plain=mcd_box_plain,
        mcd_box_dtw=mcd_box_dtw,
        mcd_box_adv_dtw=mcd_box_adv_dtw,
        useGT=useGT)

    print("====================================================")
    print("Out put the result ...")
    print("avg_mcd_plain:", avg_mcd_plain / len(loader_val), "avg_mcd_dtw:", avg_mcd_dtw / len(loader_val),
          "avg_mcd_adv_dtw:", avg_mcd_adv_dtw / len(loader_val), "eval_acc_rec_spk:",
          eval_acc_rec_spk / len(loader_val), "eval_acc_pred_spk:", eval_acc_pred_spk / len(loader_val),
          "eval_acc_rec_emo:", eval_acc_rec_emo / len(loader_val), "eval_acc_pred_emo",
          eval_acc_pred_emo / len(loader_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default="config/MovieAnimation/preprocess.yaml"
    )
    parser.add_argument("-p2", "--preprocess_config2", type=str,
                        default="config/MovieAnimation/preprocess.yaml"
                        )
    parser.add_argument("-m", "--model_config", type=str,
                        default="config/MovieAnimation/model.yaml"
                        )
    parser.add_argument(
        "-t", "--train_config", type=str,
        default="config/MovieAnimation/train.yaml"
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    preprocess_config2 = yaml.load(
        open(args.preprocess_config2, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config, preprocess_config2)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # This encoder following chenqi (V2C-Net)
    encoder_spk = VoiceEncoder().to(device)
    encoder_emo = VoiceEncoder().to(device)
    encoder_spk.eval()
    encoder_emo.eval()

    evaluate_all_valset(model, args.restore_step, configs,
                        vocoder, encoder_spk, encoder_emo)

    print("Restore_step ...", args.restore_step)





