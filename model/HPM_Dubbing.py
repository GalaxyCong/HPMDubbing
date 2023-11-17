import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Decoder, PostNet
from .modules import Affective_Prosody_Adaptor, Multi_head_Duration_Aligner, Scene_aTmos_Booster
from utils.tools import get_mask_from_lengths, generate_square_subsequent_mask
LRELU_SLOPE = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HPM_Dubbing(nn.Module):
    """ HPM_Dubbing """
    def __init__(self, preprocess_config, preprocess_config2, model_config):
        super(HPM_Dubbing, self).__init__()
        self.model_config = model_config
        self.dataset_name = preprocess_config["dataset"]
        # self.style_encoder = MelStyleEncoder(model_config)  # In fact, during conducting expriment, we remove this auxiliary style encoder (V2C-Net from Chenqi, et.al). Specifically, we only use the pre-trained GE2E model to gurateen only style information without content information, following the setting of paper. 
        self.MDA = Multi_head_Duration_Aligner(preprocess_config, model_config)
        self.APA = Affective_Prosody_Adaptor(preprocess_config, model_config)
        if self.dataset_name == "MovieAnimation":
            self.STB = Scene_aTmos_Booster(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.pre_net_bottleneck = model_config["transformer"]["pre_net_bottleneck"]
        self.postnet = PostNet()
        self.proj_fusion = nn.Conv1d(768, 256, kernel_size=1, padding=0, bias=False)
        
        self.Identity_enhancement = model_config["Enhancement"]["Identity_enhancement"]
        self.Content_enhancement = model_config["Enhancement"]["Content_enhancement"]

        self.pro_output_os = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        self.n_speaker = 1
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker = len(json.load(f))
            with open(
                    os.path.join(
                        preprocess_config2["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker += len(json.load(f))
            self.speaker_emb = nn.Embedding(
                self.n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.pro_output_emo = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        
        self.CTC_classifier_mel = CTC_classifier_mel(model_config["Symbols"]["phonemenumber"])  # len(symbols)
        self.n_emotion = 1
        self.Synchronization_coefficient = 4  
        """
        ===============================
        Q&A: Why is the Synchronization_coefficient set 4, can I change it using another positive integer? 
             Follow the Formula: n = \frac{T_{mel}}{T_v}=\frac{sr/hs}{FPS} \in \mathbb{N}^{+}.
             e.g., in our paper, for chem dataset, we set the sr == 16000Hz, hs == 160, win == 640, FPS == 25, so n is 4.
                                for chem dataset, we set the sr == 22050Hz, hs == 220, win == 880, FPS == 25, so n is 4.009. (This is the meaning of the approximately equal sign in the article). 
        
        ===============================
        Q&A: Why the use the different sr for two datasets?
             Because, we need to keep the same the expriment setting with original paper. 
             Specifically, for Chem dataset ===> NeuralDubber (https://tsinghua-mars-lab.github.io/NeuralDubber/) used the 16000Hz in their expriment. 
             for V2C dataset (chenqi et.al) ====> V2C-Net (https://github.com/chenqi008/V2C) used the 22050Hz as their result. 
             Next step, we have a plan to provide the V2C dataset (16kHz, 24KHz) Version, or Chem dataset (22050Hz, 24KHz) version.
        
        ===============================
        Q&A: Why did you provide two specialized Vocoders? Can I use the official HiFiGAN pre-trained model to replace them?
             In official HiFiGAN, sr is 22050Hz, hop_size is 256, win_size is 1024. 
             So undering this setting, we suggest to use our Vocoder to satify above Formula. 
             We have released our pre-train model (HPM_Chem, HPM_V2C), you can download it (https://github.com/GalaxyCong/HPMDubbing_Vocoder). 
        """
        if model_config["with_emotion"]:
            self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
            self.emotion_emb = nn.Embedding(
                self.n_emotion + 1,
                model_config["transformer"]["encoder_hidden"],
                padding_idx=self.n_emotion,
            )
        
        
    def forward(
            self,
            speakers,
            texts, # 3
            src_lens, # 4
            max_src_len, # 5
            mels=None,  #  6
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            spks=None,
            emotions=None,
            emos=None,
            Feature_256=None,
            lip_lens = None,
            max_lip_lens = None,
            lip_embedding = None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            useGT=None,
    ):
        """===========mask for voice, text, lip-movement========"""
        src_masks = get_mask_from_lengths(src_lens, max_src_len) 
        lip_masks = get_mask_from_lengths(lip_lens, max_lip_lens)
        if useGT:
            mel_masks = (
                get_mask_from_lengths(mel_lens, max_mel_len)
            )
        else:
            mel_masks = (
                get_mask_from_lengths(lip_lens*self.Synchronization_coefficient, max_lip_lens*self.Synchronization_coefficient)
            )

        """=========Duration Aligner========="""
        (output, ctc_pred_MDA_video) = self.MDA(lip_embedding, lip_masks, texts, src_masks, max_src_len, lip_lens, src_lens, reference_embedding = spks)

        text = output  # Hierarchical1: pronunciation information, used for concatenation
        
        """=========Add Style and emotion Vector following V2C========="""
        # if self.n_speaker > 1:
        #     if self.model_config["learn_speaker"]:
        #         output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #             -1, max_src_len, -1)
        #     else:
        #         output = output + style_vector.unsqueeze(1).expand(-1, max_mel_len, -1)  # V2C-Net
        # if self.dataset_name == "MovieAnimation":
        #     if self.n_emotion > 1:
        #         if self.model_config["learn_emotion"]:
        #             output = output + self.emotion_emb(emotions).unsqueeze(1).expand(
        #                 -1, max_src_len, -1)
        #         else:
        #             output = output + spks.unsqueeze(1).expand(-1, max_mel_len, -1)  # V2C-Net
        # In this work, we remove this add operation following V2C-Net. 
        # Identity_enhancement
        
        if self.Identity_enhancement:
            output = torch.cat([output, spks.unsqueeze(1).expand(-1, max_mel_len, -1)], dim=-1)
            output = self.pro_output_os(output.transpose(1, 2)).transpose(1, 2)
            output = output + text   
    
        """=========Prosody Adaptor========="""
        (output, p_predictions, e_predictions,) = self.APA(output, mel_masks, max_mel_len, p_targets, e_targets,
                                                        Feature_256, spks, p_control, e_control, d_control, useGT)

        prosody = output  # Hierarchical2: prosody, used for concatenation
        
        if self.Identity_enhancement:
            output = torch.cat([output, spks.unsqueeze(1).expand(-1, max_mel_len, -1)], dim=-1)
            output = self.pro_output_os(output.transpose(1, 2)).transpose(1, 2)
            output = output + prosody
            
        """=========Atmosphere Booster========="""
        if self.dataset_name == "MovieAnimation":
            # Scene feature extrated provied by V2C-Net, chenqi et.al
            E_scene = emos.unsqueeze(1).expand(-1, max_lip_lens, -1)
            (output, emotion_prediction) = self.STB(output, E_scene, mel_masks, lip_lens, max_lip_lens)  # Due to the Chem dataset not having emotional directivity or labels, this module is not used. 
            emo = output.unsqueeze(1).expand(-1, max_mel_len, -1)
            # Hierarchical3: emotion ===> output, used for concatenation
            if self.Identity_enhancement:     
                emo = torch.cat([emo, spks.unsqueeze(1).expand(-1, max_mel_len, -1)], dim=-1)
                emo = self.pro_output_emo(emo.transpose(1, 2)).transpose(1, 2)  
        else:
            emotion_prediction = None
            emo = output

        # Following the paper concatenation the three information
        fusion_output = torch.cat([text, prosody, emo], dim=-1)
        fusion_output = self.proj_fusion(fusion_output.transpose(1, 2)).transpose(1, 2)
        
        if self.Content_enhancement:
            fusion_output = fusion_output + text

        """=========Mel-Generator========="""
        fusion_output, mel_masks = self.decoder(fusion_output, mel_masks)
        ctc_pred_mel = self.CTC_classifier_mel(fusion_output)
        
        fusion_output = self.mel_linear(fusion_output)
        postnet_output = self.postnet(fusion_output) + fusion_output
        
        ctc_loss_all = [ctc_pred_MDA_video, ctc_pred_mel]

        return (
            fusion_output,
            postnet_output,
            p_predictions,
            e_predictions,
            src_masks,
            mel_masks,
            src_lens,
            lip_lens*self.Synchronization_coefficient,
            ctc_loss_all,
            emotion_prediction,
            max_src_len,
        )

class CTC_classifier_mel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, Dub):
        size = Dub.size()
        Dub = Dub.reshape(-1, size[2]).contiguous()
        Dub = self.classifier(Dub)
        return Dub.reshape(size[0], size[1], -1) 
        




