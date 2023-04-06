import os
import glob
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines


class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz', use_boundary=False):
        assert os.path.isfile( label_fp ), \
            f"File path provided for the labels does not exist. Path iput: {label_fp}."
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = True
        self.use_boundary = use_boundary
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        if self.use_boundary or (self.is_var_length and data_partition == "train"):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        self._get_files_for_partition()


        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            self.list[i] = [ x, self._labels.index( label ) ]
            self.instance_ids[i] = self._get_instance_id_from_path( x )

        print(f"Partition {self._data_partition} loaded")

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def _get_files_for_partition(self):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')
        search_str_npy = os.path.join(dir_fp, '*', self._data_partition, '*.npy')
        search_str_mp4 = os.path.join(dir_fp, '*', self._data_partition, '*.mp4')
        self._data_files.extend( glob.glob( search_str_npz ) )
        self._data_files.extend( glob.glob( search_str_npy ) )
        self._data_files.extend( glob.glob( search_str_mp4 ) )

        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [ f for f in self._data_files if f.split('/')[self.label_idx] in self._labels ]

    def load_data(self, filename):

        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print(f"Error when reading file: {filename}")
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:] )  # swap base folder
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)  

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )   # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]


    def _get_boundary(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:] )  # swap base folder
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int( utterance_duration/2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int( utterance_duration/2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length and not self.use_boundary:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        if self.use_boundary:
            boundary = self._get_boundary(self.list[idx][0], raw_data)
            return preprocess_data, label, boundary
        else:
            return preprocess_data, label

    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
    elif data_tuple[0].ndim == 3:
        max_len, h, w = data_tuple[0].shape
        data_np = np.zeros((len(data_tuple), max_len, h, w))
    for idx in range( len(data_np)):
        data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)

    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels
