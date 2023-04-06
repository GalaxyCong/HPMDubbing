from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path, action_on_extraction
from omegaconf import OmegaConf
import torch
import os
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# get video path
video_path = "/media/qichen/ea3763ef-ce45-4f0b-bf4c-646ca3ffa4d3/dataset/V2C/MovieAnimation"
video_path_list = []
for speaker in os.listdir(video_path):
	video_speaker_path = os.path.join(video_path, speaker, "00")
	for file in os.listdir(video_speaker_path):
		if os.path.splitext(file)[1]==".mp4":
			video_path_list.append(os.path.join(video_speaker_path, file))
# assert False

# Select the feature type
feature_type = 'i3d'

# max pooling
max_pool = torch.nn.MaxPool1d(3, stride=4)

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
# args.video_paths = ['./sample/v_ZNVhz7ctTq0.mp4']
args.video_paths = video_path_list
# args.show_pred = True
# args.stack_size = 24
# args.step_size = 24
args.stack_size = 12
args.step_size = 12
# args.extraction_fps = 25
args.flow_type = 'raft' # 'pwc' is not supported on Google Colab (cupy version mismatch)
args.streams = 'flow'

# Load the model
extractor = ExtractI3D(args)
model, class_head = extractor.load_model(device)

# save path
save_path = "/home/qichen/Desktop/Avatar2/V2C/preprocessed_data/MovieAnimation/emos"

# Extract features
for video_path in tqdm(args.video_paths):
	# print(f'Extracting for {video_path}')
	folder_name = video_path.split("/")[-3]
	filename = video_path.split("/")[-1].split(".")[0]

	features = extractor.extract(device, model, class_head, video_path)
	# print(features)
	# assert False
	features = torch.from_numpy(features['flow'])
	features = torch.mean(features, dim=0, keepdim=True)
	# print(features.shape)
	# assert False
	features = max_pool(features).squeeze()
	# print(features.shape)
	features = features.numpy()

	np.save("{}/{}-emo-{}.npy".format(save_path, folder_name, filename), features)
	# [(print(k), print(v.shape), print(v)) for k, v in features.items()]
	# action_on_extraction(features, video_path, output_path='./output', on_extraction='save_numpy')
