import torch
import torch.nn as nn

from .conv import ConvBlocks
from .unet import Unet


def regulate_boundary(bd_logits, threshold, min_gap=18, ref_bd=None, ref_bd_min_gap=8, non_padding=None):
    device = bd_logits.device
    bd_logits = torch.sigmoid(bd_logits).detach().cpu()
    bd = (bd_logits > threshold).long()
    bd_res = torch.zeros_like(bd).long()
    for i in range(bd.shape[0]):
        bd_i = bd[i]
        last_bd_idx = -1
        start = -1
        for j in range(bd_i.shape[0]):
            if bd_i[j] == 1:
                if 0 <= start < j:
                    continue
                if start < 0:
                    start = j
            else:
                if 0 <= start < j:
                    if j - 1 > start:
                        bd_idx = start + int(torch.argmax(bd_logits[i, start:j]).item())
                    else:
                        bd_idx = start
                    if bd_idx - last_bd_idx < min_gap and last_bd_idx > 0:
                        bd_idx = round((bd_idx + last_bd_idx) / 2)
                        bd_res[i, last_bd_idx] = 0
                    bd_res[i, bd_idx] = 1
                    last_bd_idx = bd_idx
                    start = -1

    if ref_bd is not None and ref_bd_min_gap > 0:
        ref = ref_bd.detach().cpu()
        for i in range(bd_res.shape[0]):
            ref_bd_i = ref[i]
            ref_bd_i_js = []
            for j in range(ref_bd_i.shape[0]):
                if ref_bd_i[j] == 1:
                    ref_bd_i_js.append(j)
                    seg_sum = torch.sum(bd_res[i, max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap])
                    if seg_sum == 0:
                        bd_res[i, j] = 1
                    elif seg_sum == 1 and bd_res[i, j] != 1:
                        bd_res[i, max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap] = ref_bd_i[
                            max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap
                        ]
                    elif seg_sum > 1:
                        for k in range(1, ref_bd_min_gap + 1):
                            if bd_res[i, max(0, j - k)] == 1 and ref_bd_i[max(0, j - k)] != 1:
                                bd_res[i, max(0, j - k)] = 0
                                break
                            if (
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] == 1
                                and ref_bd_i[min(bd_res.shape[1] - 1, j + k)] != 1
                            ):
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] = 0
                                break
                        bd_res[i, j] = 1
            assert torch.sum(bd_res[i, ref_bd_i_js]) == len(ref_bd_i_js), (
                f"{torch.sum(bd_res[i, ref_bd_i_js])} {len(ref_bd_i_js)}"
            )

    bd_res = bd_res.to(device)
    bd_res[:, 0] = 0
    if non_padding is not None:
        for i in range(bd_res.shape[0]):
            bd_res[i, sum(non_padding[i]) - 1 :] = 0
    else:
        bd_res[:, -1] = 0
    return bd_res


class BackboneNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hidden_size = hparams["hidden_size"]
        updown_rates = [2, 2, 2]
        channel_multiples = [1, 1, 1]
        if hparams.get("updown_rates") is not None:
            updown_rates = [int(i) for i in hparams["updown_rates"].split("-")]
        if hparams.get("channel_multiples") is not None:
            channel_multiples = [float(i) for i in hparams["channel_multiples"].split("-")]
        assert len(updown_rates) == len(channel_multiples)

        bkb_net = hparams.get("bkb_net", "conv")
        if bkb_net == "conformer":
            from .conformer import ConformerLayers

            mid_net = ConformerLayers(
                hidden_size,
                num_layers=hparams.get("bkb_layers", 12),
                kernel_size=hparams.get("conformer_kernel", 9),
                dropout=hparams.get("dropout", 0.0),
                num_heads=4,
            )
        elif bkb_net == "conv":
            mid_net = None
        else:
            raise ValueError(f"Unsupported ROSVOT backbone: {bkb_net}")

        self.net = Unet(
            hidden_size,
            down_layers=len(updown_rates),
            mid_layers=hparams.get("bkb_layers", 12),
            up_layers=len(updown_rates),
            kernel_size=3,
            updown_rates=updown_rates,
            channel_multiples=channel_multiples,
            is_BTC=True,
            constant_channels=False,
            mid_net=mid_net,
            use_skip_layer=hparams.get("unet_skip_layer", False),
        )

    def forward(self, x):
        return self.net(x)


class PitchDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hidden_size = hparams["hidden_size"]
        self.hidden_size = hidden_size
        self.pitch_attn_num_head = hparams.get("pitch_attn_num_head", 1)
        self.multihead_dot_attn = nn.Linear(hidden_size, self.pitch_attn_num_head)
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.post = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            kernel_size=3,
            layers_in_block=1,
            c_multiple=1,
            num_layers=1,
            post_net_kernel=3,
            act_type="leakyrelu",
        )
        self.pitch_out = nn.Linear(hidden_size, hparams.get("note_num", 100) + 4)
        self.note_num = hparams.get("note_num", 100)
        self.note_start = hparams.get("note_start", 30)
        self.pitch_temperature = max(1e-7, hparams.get("note_pitch_temperature", 1.0))

    def forward(self, feat, note_bd):
        bsz, _, _ = feat.shape
        attn = torch.sigmoid(self.multihead_dot_attn(feat))
        attn_feat = torch.mean(feat.unsqueeze(3) * attn.unsqueeze(2), dim=-1)
        mel2note = torch.cumsum(note_bd, 1)
        note_length = int(torch.max(torch.sum(note_bd, dim=1)).item()) + 1
        note_lengths = torch.sum(note_bd, dim=1) + 1

        attn = torch.mean(attn, dim=-1, keepdim=True)
        denom = mel2note.new_zeros(bsz, note_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2note, src=attn.squeeze(-1)
        )
        frame2note = mel2note.unsqueeze(-1).repeat(1, 1, self.hidden_size)
        note_aggregate = frame2note.new_zeros(bsz, note_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2note, src=attn_feat
        )
        note_aggregate = note_aggregate / (denom.unsqueeze(-1) + 1e-5)
        note_logits = self.pitch_out(self.post(note_aggregate)) / self.pitch_temperature

        note_pred = torch.argmax(torch.softmax(note_logits, dim=-1), dim=-1)
        note_pred[note_pred > self.note_num] = 0
        note_pred[note_pred < self.note_start] = 0
        return note_lengths, note_logits, note_pred


class MidiExtractor(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hidden_size = hparams["hidden_size"]
        self.hidden_size = hidden_size
        self.note_bd_threshold = hparams.get("note_bd_threshold", 0.5)
        self.note_bd_min_gap = round(
            hparams.get("note_bd_min_gap", 100) * hparams["audio_sample_rate"] / 1000 / hparams["hop_size"]
        )
        self.note_bd_ref_min_gap = round(
            hparams.get("note_bd_ref_min_gap", 50) * hparams["audio_sample_rate"] / 1000 / hparams["hop_size"]
        )

        self.mel_proj = nn.Conv1d(hparams["use_mel_bins"], hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            kernel_size=3,
            layers_in_block=2,
            c_multiple=1,
            num_layers=1,
            post_net_kernel=3,
            act_type="leakyrelu",
        )
        self.use_pitch = hparams.get("use_pitch_embed", True)
        if self.use_pitch:
            self.pitch_embed = nn.Embedding(300, hidden_size, padding_idx=0)
            self.uv_embed = nn.Embedding(3, hidden_size, padding_idx=0)
        self.use_wbd = hparams.get("use_wbd", True)
        if self.use_wbd:
            self.word_bd_embed = nn.Embedding(3, hidden_size, padding_idx=0)
        self.cond_encoder = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            kernel_size=3,
            layers_in_block=1,
            c_multiple=1,
            num_layers=1,
            post_net_kernel=3,
            act_type="leakyrelu",
        )
        self.net = BackboneNet(hparams)
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get("note_bd_temperature", 1.0))
        self.pitch_decoder = PitchDecoder(hparams)

    def run_encoder(self, mel=None, word_bd=None, pitch=None, uv=None):
        mel_embed = self.mel_encoder(self.mel_proj(mel.transpose(1, 2)).transpose(1, 2))
        pitch_embed = word_bd_embed = 0
        if self.use_pitch and pitch is not None and uv is not None:
            pitch_embed = self.pitch_embed(pitch) + self.uv_embed(uv)
        if self.use_wbd and word_bd is not None:
            word_bd_embed = self.word_bd_embed(word_bd)
        return self.cond_encoder(mel_embed + pitch_embed + word_bd_embed)

    def forward(self, mel=None, word_bd=None, pitch=None, uv=None, non_padding=None):
        feat = self.net(self.run_encoder(mel, word_bd, pitch, uv))
        note_bd_logits = self.note_bd_out(feat).squeeze(-1) / self.note_bd_temperature
        note_bd_logits = torch.clamp(note_bd_logits, min=-16.0, max=16.0)
        note_bd = regulate_boundary(
            note_bd_logits,
            self.note_bd_threshold,
            self.note_bd_min_gap,
            word_bd,
            self.note_bd_ref_min_gap,
            non_padding,
        )
        note_lengths, note_logits, note_pred = self.pitch_decoder(feat, note_bd)
        return {
            "note_bd_logits": note_bd_logits,
            "note_bd_pred": note_bd,
            "note_lengths": note_lengths,
            "note_logits": note_logits,
            "note_pred": note_pred,
        }
