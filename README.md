# HPMDubbing - PyTorch Implementation

In this [paper](), we propose a novel movie dubbing architecture via hierarchical prosody modeling, which bridges the visual information to corresponding speech prosody from three aspects: lip, face, and scene. Specifically, we align lip movement to the speech duration, and convey facial expression to speech energy and pitch via attention mechanism based on valence and arousal representations inspired by the psychology findings. Moreover, we design an emotion booster to capture the atmosphere from global video scenes. All these embeddings are used together to generate mel-spectrogram, which is then converted into speech waves by an existing vocoder. Extensive experimental results on the V2C and Chem benchmark datasets demonstrate the favourable performance of the proposed method.

[//]: # (We provide our implementation and pre-trained models as open-source in this repository. )

[//]: # (&#40;Continue to upload, before the upload is finished, don't rush to run 🌟&#41;)

Visit our [demo website]() or [download the dubbing samples]() to see more results.
# Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Dataset

[V2C-MovieAnimation](https://github.com/chenqi008/V2C) is a multi-speaker dataset for animation movie dubbing with identity and emotion annotations. It is collected from 26 Disney cartoon movies and covers 153 diverse characters.

In this work, we release the [V2C-MovieAnimation2.0](https://pan.baidu.com/s/1UdNjEytLyUxy60xVJoXgkA) (password: good) to satisfy the requirement of dubbing the specified characters. 
Specifically, we removed redundant character faces in movie frames (please note that our video frames are sampled at 25 FPS by ffmpeg). 
You can download our preprocessed features directly through the link.
![Illustration](./images/Our_V2C2.0_Illustration.jpeg)

# Data Preparation

For voice preprocessing (mel-spectrograms, pitch, and energy), Montreal Forced Aligner (MFA) is used to obtain the alignments between the utterances and the phoneme sequences. Alternatively, you can skip the below-complicated step, and use our extracted features, directly.

Download the official [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) package and run
```
./montreal-forced-aligner/bin/mfa_align /data/conggaoxiang/HPMDubbing/V2C_Data/wav16 /data/conggaoxiang/HPMDubbing/lexicon/librispeech-lexicon.txt  english /data/conggaoxiang/HPMDubbing/V2C_Code/example_V2C16/TextGrid -j
```
then, please run the below script to save the .npy files of mel-spectrograms, pitch, and energy from two datasets, respectively.
```
python V2C_preprocess.py config/MovieAnimation/preprocess.yaml
```
```
python Chem_preprocess.py config/MovieAnimation/preprocess.yaml
```
For hierarchical visual feature preprocessing (lip, face, and scenes), we detect and crop the face from the video frames using $S^3FD$ [face detection model](https://github.com/yxlijun/S3FD.pytorch). Then, we align faces to generate 68 landmarks and bounding boxes (./landmarks and ./boxes). Finally, we get the mouth ROIs from all video clips, following [EyeLipCropper](https://github.com/zhliuworks/EyeLipCropper). Similarly, you can also skip the complex steps below and directly use the features we extracted.

# Training

For V2C-MovieAnimation dataset, please run train.py file with
```
python train.py -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
```
For Chem dataset, please run train.py file with
```
python train.py -p config/Chem/preprocess.yaml -m config/Chem/model.yaml -t config/Chem/train.yaml -p2 config/Chem/preprocess.yaml
```
![Illustration](./images/train.jpeg)
# Inferrence
```
python Inference.py --restore_step [Chekpoint] -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
```


# Tensorboard
Use
```
tensorboard --logdir output/log/MovieAnimation --port=xxxx
```
or 
```
tensorboard --logdir output/log/Chem --port=xxxx
```
to serve TensorBoard on your localhost.
The loss curves, mcd curves, synthesized mel-spectrograms, and audios are shown.


# References
- [V2C: Visual Voice Cloning](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_V2C_Visual_Voice_Cloning_CVPR_2022_paper.pdf), Q. Chen, *et al*.
- [Neural Dubber: Dubbing for Videos According to Scripts](https://proceedings.neurips.cc/paper/2021/file/8a9c8ac001d3ef9e4ce39b1177295e03-Paper.pdf), C. Hu, *et al*.
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.

# Citation
