import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_result(preprocess_config2, model_config, model, vocoder, loader, sampling_rate=None, samples_path=None, useGT=False):
    # Evaluation
    counter_batch = 0

    for batchs in tqdm(loader):
        wav_reconstructions_batch = []
        wav_predictions_batch = []
        tags_batch =[]
        speakers_batch = []
        emotions_batch = []
        cofs_batch = []
        counter_batch+=1
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
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
        save_wav(sampling_rate, samples_path,
                   wav_reconstructions_batch, wav_predictions_batch, tags_batch)




def Inference_wav(model, step, configs, vocoder=None):
    preprocess_config, model_config, train_config, preprocess_config2 = configs
    useGT = False

    val_samples_path = train_config["path"]["result_path"]
    sampling_rate = preprocess_config2["preprocessing"]["audio"]["sampling_rate"]

    dataset_val = Dataset(
        "val.txt", preprocess_config2, train_config, sort=False, drop_last=False, diff_audio=True
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset_val.collate_fn,
    )

    print("Start load all val-set", '\n')
    print('The number of the val-set:', len(dataset_val), '\n')
    # Get loss function
    # Loss = HPM_DubbingLoss(preprocess_config2, model_config).to(device)
    generate_result(preprocess_config2, model_config, model, vocoder,loader_val, sampling_rate=sampling_rate, samples_path=val_samples_path, useGT=useGT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=420000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument("-p2", "--preprocess_config2", type=str,
                        required=True, help="path to the second preprocess.yaml",
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
    preprocess_config2 = yaml.load(
        open(args.preprocess_config2, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config, preprocess_config2)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    # val_samples_path = "./output/result/MovieAnimation"
    print("Generating wav...")
    Inference_wav(model, args.restore_step, configs, vocoder)
    # print("All Done")
    






