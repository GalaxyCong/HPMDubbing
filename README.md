# HPMDubbing - PyTorch Implementation

This is a PyTorch implementation of [**Learning to Dub Movies via Hierarchical Prosody Models**].


# Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Dataset
V2C-MovieAnimation is a multi-speaker dataset for animation movie dubbing. It is collected from 26 Disney cartoon movies and covers 153 diverse characters.
The link of raw data can be found from [here](https://pan.baidu.com/s/1wbmd4HnpDsLjTn0YbwwASA), password: eewv

In our work, we release the [V2C-MovieAnimation2.0](https://pan.baidu.com/s/1wbmd4HnpDsLjTn0YbwwASA), password: ectr. To satisfy the requirement of dubbing the specified characters, we removed redundant character faces in movie frames.
 (Please note that our video frames are sampled at 25 FPS by ffmpeg.)
![img.png](img.png)
# Data Preparation
