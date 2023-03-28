import torch
import torchaudio

import os
from tqdm import tqdm
import numpy as np

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

# get video path
audio_path = "/media/qichen/ea3763ef-ce45-4f0b-bf4c-646ca3ffa4d3/dataset/V2C/MovieAnimation"
audio_path_list = []
for speaker in os.listdir(audio_path):
	audio_speaker_path = os.path.join(audio_path, speaker, "00")
	for file in os.listdir(audio_speaker_path):
		if os.path.splitext(file)[1]==".wav":
			audio_path_list.append(os.path.join(audio_speaker_path, file))

# save path
save_path = "/home/qichen/Desktop/Avatar2/V2C/preprocessed_data/MovieAnimation/spk2"

# extract feature
for audio_path in tqdm(audio_path_list):
	folder_name = audio_path.split("/")[-3]
	filename = audio_path.split("/")[-1].split(".")[0]

	# wav_tensor, sample_rate = torchaudio.load("BossBaby-00-0196.wav")
	wav_tensor, sample_rate = torchaudio.load(audio_path)
	mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
	emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
	emb_tensor = emb_tensor.detach().numpy()

	np.save("{}/{}-spk-{}.npy".format(save_path, folder_name, filename), emb_tensor)



