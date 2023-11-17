import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, synth_multi_samples
from model import HPM_DubbingLoss
from dataset import Dataset

import numpy as np

from scipy.io.wavfile import write
from tqdm import tqdm
import sys

sys.path.append("..")
from resemblyzer import preprocess_wav

from mcd import Calculate_MCD

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
        # for i in range(len(wav_reconstructions_utterance_embeds)):
        #     wav_reconstructions_utterance_embeds[i] = wav_reconstructions_utterance_embeds[i] / \
        #     (np.linalg.norm(wav_reconstructions_utterance_embeds[i], ord=2) + 1e-5)
        #     wav_predictions_utterance_embeds[i] = wav_predictions_utterance_embeds[i] / \
        #     (np.linalg.norm(wav_predictions_utterance_embeds[i], ord=2) + 1e-5)
    else:
        centroids_rec = centroids

    # similarity matrix: wav_pred 512x256; centroids: num_speaker(128)x256
    # sim_matrix_pred = np.dot(wav_predictions_utterance_embeds, centroids_rec.T) \
    #     * encoder.similarity_weight.item() + encoder.similarity_bias.item()
    # sim_matrix_rec = np.dot(wav_reconstructions_utterance_embeds, centroids_rec.T) \
    #     * encoder.similarity_weight.item() + encoder.similarity_bias.item()
    sim_matrix_pred = np.dot(wav_predictions_utterance_embeds, centroids_rec.T)
    sim_matrix_rec = np.dot(wav_reconstructions_utterance_embeds, centroids_rec.T)
    # pred_locs 512x1
    pred_locs = sim_matrix_pred.argmax(axis=1)
    rec_locs = sim_matrix_rec.argmax(axis=1)
    correct_num_pred = 0
    correct_num_rec = 0
    for i in range(len(pred_locs)):
        if loc2ids_map[pred_locs[i]] == speakers_all[i].item():
            correct_num_pred += 1
        if loc2ids_map[rec_locs[i]] == speakers_all[i].item():
            correct_num_rec += 1
    #
    eval_acc_pred = correct_num_pred / float(len(pred_locs))
    eval_acc_rec = correct_num_rec / float(len(pred_locs))

    return eval_acc_rec, eval_acc_pred


def calculate_acc(preprocess_config2, model_config, model, vocoder, \
                  encoder_spk, encoder_emo, loader, logger, sampling_rate=None, samples_path=None, \
                  mcd_box_plain=None, mcd_box_dtw=None, mcd_box_adv_dtw=None, useGT=False):
    # Evaluation

    quick_eval = True  # evaluate only 32 samples if True, otherwise evaluate all data
    counter_batch = 0

    for batchs in tqdm(loader):
        wav_reconstructions_batch = []
        wav_predictions_batch = []
        tags_batch = []
        speakers_batch = []
        emotions_batch = []
        cofs_batch = []
        counter_batch += 1
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]), useGT=useGT)

                if logger is not None:
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
        acc_means_spk, acc_means_emo, avg_mcd = assess_spk_emo(
            encoder_spk=encoder_spk, encoder_emo=encoder_emo, sampling_rate=sampling_rate, samples_path=samples_path,
            mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw,
            wav_reconstructions_batch=wav_reconstructions_batch, wav_predictions_batch=wav_predictions_batch,
            tags_batch=tags_batch, speakers_batch=speakers_batch, emotions_batch=emotions_batch, cofs_batch=cofs_batch)
        if counter_batch == 1:
            acc_sums_spk = acc_means_spk
            acc_sums_emo = acc_means_emo
            sum_mcd = avg_mcd
        else:
            acc_sums_spk = list(map(lambda x: x[0] + x[1], zip(acc_sums_spk, acc_means_spk)))
            acc_sums_emo = list(map(lambda x: x[0] + x[1], zip(acc_sums_emo, acc_means_emo)))
            sum_mcd = list(map(lambda x: x[0] + x[1], zip(sum_mcd, avg_mcd)))
        """
        V2C-Net's quick_eval setting (from chenqi et.al ). If you wanna test all data, please delete the following code.
        """
        if counter_batch == 2:
            break
    acc_sums_spk = list(np.array(acc_sums_spk) / counter_batch)
    acc_sums_emo = list(np.array(acc_sums_emo) / counter_batch)
    sum_mcd = list(np.array(sum_mcd) / counter_batch)
    return batch, output, acc_sums_spk, acc_sums_emo, sum_mcd


def assess_spk_emo(encoder_spk, encoder_emo, sampling_rate, samples_path,
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

    # reload
    print("Reloading ...")
    rec_wavs = np.array(list(map(preprocess_wav, tqdm(rec_fpaths, "Preprocessing rec wavs", len(rec_fpaths)))))
    pred_wavs = np.array(list(map(preprocess_wav, tqdm(pred_fpaths, "Preprocessing pred wavs", len(pred_fpaths)))))

    # mcd
    print("calculate MCD ...")
    for i in tqdm(range(len(rec_fpaths))):
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
    # centroids_emo = np.load("/mnt/cephfs/home/chenqi/workspace/Project/Resemblyzer/centroids_emo_all.npy")
    # centroids_emo = np.load("/home/qichen/Desktop/Avatar2/V2C/centroids_emo_all.npy")
    eval_acc_rec_emo, eval_acc_pred_emo = acc_metric(emotions_ids, emotions_batch, \
                                                     wav_reconstructions_utterance_embeds_emo,
                                                     wav_predictions_utterance_embeds_emo, \
                                                     ids2loc_map_emo, loc2ids_map_emo, centroids=None)
    # speaker
    eval_acc_rec_spk, eval_acc_pred_spk = acc_metric(speakers_ids, speakers_batch, \
                                                     wav_reconstructions_utterance_embeds_spk,
                                                     wav_predictions_utterance_embeds_spk, \
                                                     ids2loc_map, loc2ids_map)

    acc_means_spk = [eval_acc_rec_spk, eval_acc_pred_spk]
    acc_means_emo = [eval_acc_rec_emo, eval_acc_pred_emo]
    avg_mcd = [avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw]

    return acc_means_spk, acc_means_emo, avg_mcd


def evaluate(model, step, configs, logger=None, vocoder=None, encoder_spk=None, \
             encoder_emo=None, train_samples_path=None, val_samples_path=None, useGT=False):
    # preprocess_config, model_config, train_config = configs
    preprocess_config, model_config, train_config, preprocess_config2 = configs

    # Get dataset
    dataset_train = Dataset(
        "train.txt", preprocess_config2, train_config, sort=False, drop_last=False, diff_audio=True
    )
    dataset_val = Dataset(
        "val.txt", preprocess_config2, train_config, sort=False, drop_last=False, diff_audio=True
    )

    #
    loader_train = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
    )
    loader_train_acconly = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset_train.collate_fn,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset_val.collate_fn,
    )

    print("=============================")
    print("dataset_train:", len(dataset_train))
    print("loader_train:", len(loader_train))
    print("dataset_val:", len(dataset_val))
    print("loader_val:", len(loader_val))

    # sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    sampling_rate = preprocess_config2["preprocessing"]["audio"]["sampling_rate"]
    batch_size = train_config["optimizer"]["batch_size"]
    loss_model = model_config["loss_function"]["model"]
    print("calculate training acc ...")
    # initialize MCD module
    mcd_box_plain = Calculate_MCD("plain", sr=sampling_rate)
    mcd_box_dtw = Calculate_MCD("dtw", sr=sampling_rate)
    mcd_box_adv_dtw = Calculate_MCD("adv_dtw", sr=sampling_rate)
    #
    _, _, acc_means_train_spk, acc_means_train_emo, avg_mcd_train = calculate_acc(preprocess_config2, model_config, \
                                                                                  model, vocoder, encoder_spk,
                                                                                  encoder_emo, loader_train_acconly,
                                                                                  logger, sampling_rate=sampling_rate,
                                                                                  samples_path=train_samples_path, \
                                                                                  mcd_box_plain=mcd_box_plain,
                                                                                  mcd_box_dtw=mcd_box_dtw,
                                                                                  mcd_box_adv_dtw=mcd_box_adv_dtw,
                                                                                  useGT=useGT)

    print("calculate val loss and acc ...")
    # initialize MCD module
    mcd_box_plain = Calculate_MCD("plain", sr=sampling_rate)
    mcd_box_dtw = Calculate_MCD("dtw", sr=sampling_rate)
    mcd_box_adv_dtw = Calculate_MCD("adv_dtw", sr=sampling_rate)
    #
    batch, output, acc_means_val_spk, acc_means_val_emo, avg_mcd_val = calculate_acc(preprocess_config2, \
                                                                                     model_config, model, vocoder,
                                                                                     encoder_spk, encoder_emo,
                                                                                     loader_val, logger,
                                                                                     sampling_rate=sampling_rate,
                                                                                     samples_path=val_samples_path, \
                                                                                     mcd_box_plain=mcd_box_plain,
                                                                                     mcd_box_dtw=mcd_box_dtw,
                                                                                     mcd_box_adv_dtw=mcd_box_adv_dtw,
                                                                                     useGT=useGT)

    message = "Validation Step {}, MCD plain|dtw|adv_dtw (train): {:.4f}|{:.4f}|{:.4f}, \
                MCD plain|dtw|adv_dtw (val): {:.4f}|{:.4f}|{:.4f}, \
                Id. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
                Id. Acc (val) (rec|pred): {:.4f}|{:.4f}, \
                Emo. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
                Emo. Acc (val) (rec|pred): {:.4f}|{:.4f}".format(step,
                                                                 avg_mcd_train[0], avg_mcd_train[1], avg_mcd_train[2], \
                                                                 avg_mcd_val[0], avg_mcd_val[1], avg_mcd_val[2], \
                                                                 acc_means_train_spk[0], acc_means_train_spk[1], \
                                                                 acc_means_val_spk[0], acc_means_val_spk[1], \
                                                                 acc_means_train_emo[0], acc_means_train_emo[1], \
                                                                 acc_means_val_emo[0], acc_means_val_emo[1]
                                                                 )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config2,
        )
        log(logger, step, losses=None, accs_val_spk=acc_means_val_spk, \
            accs_train_spk=acc_means_train_spk, accs_val_emo=acc_means_val_emo, \
            accs_train_emo=acc_means_train_emo, avg_mcd_val=avg_mcd_val, \
            avg_mcd_train=avg_mcd_train, LM=loss_model)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
  
