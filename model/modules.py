import os
import json
import copy
import math
from collections import OrderedDict
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from utils.tools import init_weights, get_padding
from transformer import Encoder, Lip_Encoder
LRELU_SLOPE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CTC_classifier_MDA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # B, S, 512
        size = x.size()
        x = x.reshape(-1, size[2]).contiguous()
        x = self.classifier(x)
        return x.reshape(size[0], size[1], -1)  
    
class Multi_head_Duration_Aligner(nn.Module):
    """Multi_head_Duration_Aligner"""
    def __init__(self, preprocess_config, model_config):
        super(Multi_head_Duration_Aligner, self).__init__()
        self.dataset_name = preprocess_config["dataset"]
        self.encoder = Encoder(model_config)
        self.lip_encoder = Lip_Encoder(model_config)
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.attn_text_spk = nn.MultiheadAttention(256, 8, dropout=0.1)
        
        self.num_upsamples = len(model_config["upsample_ConvTranspose"]["upsample_rates"])
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(model_config["upsample_ConvTranspose"]["upsample_rates"],
                                       model_config["upsample_ConvTranspose"]["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(
                ConvTranspose1d(model_config["upsample_ConvTranspose"]["upsample_initial_channel"],
                                model_config["upsample_ConvTranspose"]["upsample_initial_channel"], k,
                                u, padding=(u // 2 + u % 2), output_padding=u % 2)))
               
        self.proj_con = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)
        
        self.CTC_classifier_MDA = CTC_classifier_MDA(model_config["Symbols"]["phonemenumber"])  # len(symbols)
        

    def forward(
            self,
            lip_embedding,
            lip_masks,
            texts,
            src_masks,
            max_src_len,
            lip_lens, 
            src_lens,
            reference_embedding = None,
    ):
        output_lip = self.lip_encoder(lip_embedding, lip_masks)
        output_text = self.encoder(texts, src_masks)
        
        # Before calculating attention between phoneme and lip-motion sequence, the text information will be fused with the speaker identity, following the paper.
        sss = reference_embedding.unsqueeze(1).expand(-1, max_src_len, -1)
        contextual_sss, _ = self.attn_text_spk(query=output_text.transpose(0, 1), key=sss.transpose(0, 1),
                                        value=sss.transpose(0, 1), key_padding_mask=src_masks)

        contextual_sss = contextual_sss.transpose(0,1)
        output_text = contextual_sss + output_text
        
        output, _ = self.attn(query=output_lip.transpose(0, 1), key=output_text.transpose(0, 1),
                                        value=output_text.transpose(0, 1), key_padding_mask=src_masks)
        output = output.transpose(0,1)
        
        output = self.proj_con(output.transpose(1, 2))

        # In our implementation, we use the CTC as an auxiliary loss to help the text-video context sequence aligning information, like the diagonal alignment constraint loss on the attention output matrix in NeuralDubber (https://tsinghua-mars-lab.github.io/NeuralDubber/).  
        B = texts.shape[0] 
        ctc_pred_MDA_video = self.CTC_classifier_MDA(output.transpose(1, 2))
        
        # video length to mel length
        for i in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.ups[i](output)
        output = output.transpose(1, 2)

        return (output, ctc_pred_MDA_video)


class Affective_Prosody_Adaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(Affective_Prosody_Adaptor, self).__init__()
        self.proj_con = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        
        self.dataset_name = preprocess_config["dataset"]
        self.emo_fc_2_val = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )
        self.emo_fc_2_aro = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )

        self.W = nn.Linear(256, 256)
        self.Uo = nn.Linear(256, 256)
        self.Um = nn.Linear(256, 256)

        self.bo = nn.Parameter(torch.ones(256), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(256), requires_grad=True)

        self.wo = nn.Linear(256, 1)
        self.wm = nn.Linear(256, 1)
        self.inf = 1e5

        self.loss_model = model_config["loss_function"]["model"]

        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.scale_fusion = model_config["Affective_Prosody_Adaptor"]["Use_Scale_attention"]
        self.predictor_ = model_config["variance_predictor"]["predictor"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control, useGT):
        prediction = self.pitch_predictor(x, mask)  # prediction for each src frame
        if useGT:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, useGT):
        prediction = self.energy_predictor(x, mask)
        if useGT:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )

        return prediction, embedding

    def forward(
            self,
            x,
            mel_mask=None,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            Feature_256=None,
            spks=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            useGT=None,
    ):
        M = x
        valence = self.emo_fc_2_val(Feature_256)
        if self.scale_fusion:
            context_valence, _ = self.arousal_attention(M, valence, valence)  # torch.Size([32, 448, 256])
        else:
            sample_numb = valence.shape[1]
            W_f2d = self.W(M)
            U_objs = self.Uo(valence)
            attn_feat_V = W_f2d.unsqueeze(2) + U_objs.unsqueeze(
                1) + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
            attn_weights_V = self.wo(torch.tanh(attn_feat_V))  # (bsz, sample_numb, max_objects, 1)
            objects_mask_V = mel_mask[:, None, :, None].repeat(1, sample_numb, 1, 1).permute(0,2,1,3)  # (bsz, sample, max_objects_per_video, 1)
            attn_weights_V = attn_weights_V - objects_mask_V.float() * self.inf
            attn_weights_V = attn_weights_V.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
            attn_objects_V = attn_weights_V * attn_feat_V
            context_valence = attn_objects_V.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            context_valence, pitch_target, mel_mask, p_control, useGT
        )
        pitch_M = M + pitch_embedding

        Arousal = self.emo_fc_2_aro(Feature_256)
        if self.scale_fusion:
            context_arousal, _ = self.arousal_attention(M, Arousal, Arousal)
        else:
            sample_numb = Arousal.shape[1]
            W_f2d = self.W(M)
            U_motion = self.Um(Arousal)
            attn_feat = W_f2d.unsqueeze(2) + U_motion.unsqueeze(
                1) + self.bm  # (bsz, sample_numb, max_objects, hidden_dim)
            attn_weights = self.wm(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
            objects_mask = mel_mask[:, None, :, None].repeat(1, sample_numb, 1, 1).permute(0, 2, 1,3)  # (bsz, sample, max_objects_per_video, 1)
            attn_weights = attn_weights - objects_mask.float() * self.inf
            attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
            attn_objects = attn_weights * attn_feat
            context_arousal = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
        energy_prediction, energy_embedding = self.get_energy_embedding(
            context_arousal, energy_target, mel_mask, e_control, useGT
        )
        energy_M = M + energy_embedding

        # concatenation the energy and pitch information, following the paper
        prosody = torch.cat([pitch_M, energy_M], dim=-1)
        prosody = self.proj_con(prosody.transpose(1, 2)).transpose(1, 2)
    
        return (
            prosody,
            pitch_prediction,
            energy_prediction,
        )


class Scene_aTmos_Booster(nn.Module):
    """Multi_head_Duration_Aligner"""
    def __init__(self, preprocess_config, model_config):
        super(Scene_aTmos_Booster, self).__init__()
        self.dataset_name = preprocess_config["dataset"]
        self.emo_fc_2_sence = nn.Sequential(nn.Linear(256, 128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 256),
                                            nn.ReLU(inplace=True),
                                            )
        self.Emo_attention = nn.MultiheadAttention(256, 4, dropout=0.1)
        
        self.ifdown_emotion = 1
        
        self.emo_classifier = AdversarialClassifier(
            in_dim=256,
            out_dim=8,
            hidden_dims=model_config["classifier"]["cls_hidden"]
        )

    def forward(
            self,
            output,
            emos,
            mel_mask,
            lip_lens, max_lip_lens,
    ):
        emos = self.emo_fc_2_sence(emos)
        emo_context, _ = self.Emo_attention(query=emos.transpose(0, 1), key=output.transpose(0, 1),
                                        value=output.transpose(0, 1), key_padding_mask=mel_mask)
        emo_context = emo_context.transpose(0,1)
        
        # self.get_Emo_embedding
        emo_lens = lip_lens // self.ifdown_emotion
        emo_lens[emo_lens == 0] = 1
        max_emo_lens = max_lip_lens // self.ifdown_emotion
        emo_masks = (1 - get_mask_from_lengths(emo_lens, max_emo_lens).float()).unsqueeze(-1).expand(-1, -1, 256)
        trained_speakerembedding = torch.sum(emo_context * emo_masks, axis=1) / emo_lens.unsqueeze(-1).expand(-1, 256)
        emotion_prediction = self.emo_classifier(trained_speakerembedding, is_reversal=False)

        return (trained_speakerembedding, emotion_prediction)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function.
    In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer
    Code from:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    
class AdversarialClassifier(nn.Module):
    """
    AdversarialClassifier
        - 1 gradident reversal layer
        - n hidden linear layers with ReLU activation
        - 1 output linear layer with Softmax activation
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[256], rev_scale=1):
        """
        Args:
            in_dim: input dimension
            out_dim: number of units of output layer (number of classes)
            hidden_dims: number of units of hidden layers
            rev_scale: gradient reversal scale
        """
        super(AdversarialClassifier, self).__init__()

        self.gradient_rev = GradientReversal(rev_scale)

        in_sizes = [in_dim] + hidden_dims[:]
        out_sizes = hidden_dims[:] + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.activations = [nn.ReLU()] * len(hidden_dims) + [nn.Softmax(dim=-1)]

    def forward(self, x, is_reversal=True):
        if is_reversal:
            x = self.gradient_rev(x)
        for (linear, activate) in zip(self.layers, self.activations):
            x = activate(linear(x))
        return x
    

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x



