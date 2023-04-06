import os
import json

import torch
import numpy as np

from utils.env import AttrDict
from utils.hifigan_16_models import Generator
from utils.istft_models import istft_Generator

from utils.stft import TorchSTFT
from model import HPM_Dubbing, ScheduledOptim

MAX_WAV_VALUE = 32768.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args, configs, device, train=False):
    # (preprocess_config, model_config, train_config) = configs
    (preprocess_config, model_config, train_config, preprocess_config2) = configs
    model = HPM_Dubbing(preprocess_config, preprocess_config2, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        # remove keys of pretrained model that are not in our model (i.e., embeeding layer)
        model_dict = model.state_dict()
        if model_config["learn_speaker"]:
            speaker_emb_weight = ckpt["model"]["speaker_emb.weight"]
            s, d = speaker_emb_weight.shape
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() \
                         if k in model_dict and k != "speaker_emb.weight"}
        model.load_state_dict(ckpt["model"], strict=False)
        if model_config["learn_speaker"] and s <= model.state_dict()["speaker_emb.weight"].shape[0]:
            model.state_dict()["speaker_emb.weight"][:s, :] = speaker_emb_weight

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]  # HiFi_GAN_16
    speaker = config["vocoder"]["speaker"]  # LJSpeech_16KHz
    checkpoint_path = config["vocoder"]["vocoder_checkpoint_path"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        config_file = '/data/conggaoxiang/vocoder/hifi-gan-master/checkpoint_hifigan_offical/config.json'
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(
            '/data/conggaoxiang/vocoder/hifi-gan-master/checkpoint_hifigan_offical/generator_v1',
            device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    elif name == "HiFi_GAN_16":
        config_file = os.path.join(checkpoint_path, "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)

        vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(os.path.join(checkpoint_path, "g_HPM_Chem"),
                                       device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    elif name == "HiFi_GAN_220":
        config_file = os.path.join(checkpoint_path, "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(os.path.join(checkpoint_path, "g_HPM_V2C"),
                                       device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    elif name == "ISTFTNET":
        config_file = '/data/conggaoxiang/vocoder/iSTFTNet-pytorch-master/cp_hifigan/checkpoint_iSTFTNet/config.json'
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        vocoder = istft_Generator(h).to(device)
        state_dict_g = load_checkpoint(
            "/data/conggaoxiang/vocoder/iSTFTNet-pytorch-master/cp_hifigan/checkpoint_iSTFTNet/g_00810000", device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    return vocoder


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]  # HiFi_GAN_16
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)  # torch.Size([1, 80, 448])
        elif name == "HiFi_GAN_16":
            wavs = vocoder(mels).squeeze(1)
        elif name == "HiFi_GAN_220":
            wavs = vocoder(mels).squeeze(1)
        elif name == "ISTFTNET":
            stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16).to(device)
            spec, phase = vocoder(mels)
            y_g_hat = stft.inverse(spec, phase)
            wavs = y_g_hat.squeeze(1)

    wavs = (
            wavs.cpu().numpy()
            * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
