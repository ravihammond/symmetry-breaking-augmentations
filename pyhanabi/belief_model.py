# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

torch.set_printoptions(
    linewidth=600
)


class V0BeliefModel(torch.jit.ScriptModule):
    def __init__(self, device, num_sample):
        super().__init__()
        self.device = device
        self.hand_size = 5
        self.bit_per_card = 25
        self.num_sample = num_sample

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        """dummy function"""
        shape = (1, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def observe(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("observe")
        return obs

    @torch.jit.script_method
    def sample(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("sample")
        v0 = obs["v0"]
        bsize = v0.size(0)
        v0 = v0.view(bsize, self.hand_size, -1)[:, :, : self.bit_per_card]

        v0 = v0.view(-1, self.bit_per_card).clamp(min=1e-5)
        sample = v0.multinomial(self.num_sample, replacement=True)
        # smaple: [bsize * handsize, num_sample]
        sample = sample.view(bsize, self.hand_size, self.num_sample)
        sample = sample.transpose(1, 2)
        return {"sample": sample, "h0": obs["h0"], "c0": obs["c0"]}


def pred_loss(logp, gtruth, seq_len):
    """
    logit: [seq_len, batch, hand_size, bits_per_card]
    gtruth: [seq_len, batch, hand_size, bits_per_card]
        one-hot, can be all zero if no card for that position
    """
    assert logp.size() == gtruth.size()
    logp = (logp * gtruth).sum(3)
    hand_size = gtruth.sum(3).sum(2).clamp(min=1e-5)
    logp_per_card = logp.sum(2) / hand_size
    xent = -logp_per_card.sum(0)
    # print(seq_len.size(), xent.size())
    assert seq_len.size() == xent.size()
    avg_xent = xent / seq_len
    nll_per_card = -logp_per_card
    return xent, avg_xent, nll_per_card

# def pred_loss(logp, gtruth, seq_len):
    # """
    # logit: [seq_len, batch, hand_size, bits_per_card]
    # gtruth: [seq_len, batch, hand_size, bits_per_card]
        # one-hot, can be all zero if no card for that position
    # """
    # assert logp.size() == gtruth.size()
    # logp = (logp * gtruth).sum(3)
    # hand_size = gtruth.sum(3).sum(2).clamp(min=1e-5)
    # logp_per_card = logp.sum(2) / hand_size
    # xent = -logp_per_card.sum(0)
    # # print(seq_len.size(), xent.size())
    # assert seq_len.size() == xent.size()
    # avg_xent = xent / seq_len
    # nll_per_card = -logp_per_card
    # return xent, avg_xent, nll_per_card


class ARBeliefModel(torch.jit.ScriptModule):
    def __init__(
        self, 
        device, 
        in_dim, 
        hid_dim, 
        hand_size, 
        out_dim, 
        num_sample, 
        fc_only, 
        parameterized,
        num_conventions,
        sad_legacy=False,
    ):
        """
        mode: priv: private belief prediction
              publ: public/common belief prediction
        """
        super().__init__()
        self.device = device
        self.input_key = "priv_s"
        self.ar_input_key = "own_hand_ar_in"
        self.convention_idx_key = "convention_idx"
        self.ar_target_key = "own_hand"
        if sad_legacy:
            self.ar_target_key = "own_hand_ar"

        self.in_dim = in_dim
        self.hand_size = hand_size
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.num_sample = num_sample
        self.fc_only = fc_only

        self.parameterized = parameterized
        self.num_conventions = num_conventions

        if self.parameterized:
            self.in_dim += self.num_conventions

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.emb = nn.Linear(25, self.hid_dim // 8, bias=False)
        self.auto_regress = nn.LSTM(
            self.hid_dim + self.hid_dim // 8,
            self.hid_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self.auto_regress.flatten_parameters()

        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @classmethod
    def load(
            cls, 
            weight_file, 
            device, 
            hand_size, 
            num_sample, 
            fc_only,
            parameterized,
            num_conventions,
            sad_legacy=False):
        state_dict = torch.load(weight_file)
        hid_dim, in_dim = state_dict["net.0.weight"].size()
        if parameterized:
            in_dim -= num_conventions
        out_dim = state_dict["fc.weight"].size(0)
        model = cls(
                device, 
                in_dim, 
                hid_dim, 
                hand_size, 
                out_dim, 
                num_sample, 
                fc_only,
                parameterized,
                num_conventions,
                sad_legacy=sad_legacy)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model

    @torch.jit.script_method
    def forward(self, x, ar_card_in):
        # x = batch.obs[self.input_key]
        x = self.net(x)
        if self.fc_only:
            o = x
        else:
            o, (h, c) = self.lstm(x)

        # ar_card_in  = batch.obs[self.ar_input_key]
        seq, bsize, _ = ar_card_in.size()
        ar_card_in = ar_card_in.view(seq * bsize, self.hand_size, 25)

        ar_emb_in = self.emb(ar_card_in)
        # ar_card_in: [seq * batch, 5, 64]
        # o: [seq, batch, 512]
        o = o.view(seq * bsize, self.hid_dim)
        o = o.unsqueeze(1).expand(seq * bsize, self.hand_size, self.hid_dim)
        ar_in = torch.cat([ar_emb_in, o], 2)
        ar_out, _ = self.auto_regress(ar_in)

        logit = self.fc(ar_out)
        logit = logit.view(seq, bsize, self.hand_size, -1)
        return logit

    def loss(self, batch, beta=1, convention_index_override=None):
        x = batch.obs[self.input_key]

        # Append convention one-hot vectors if model is parameterized
        if self.parameterized:
            # Concatenate convention one-hot vectors to x
            convention_indexes = batch.obs[self.convention_idx_key].clone()
            if convention_index_override is not None:
                convention_indexes[:,:] = convention_index_override
            one_hot = F.one_hot(convention_indexes, num_classes=self.num_conventions)
            cat_x = torch.cat((x, one_hot), 2)

            # Create mask to zero out one-hot vectors after sequences finish
            seq_list = torch.arange(cat_x.size(0)).to(batch.seq_len.get_device())
            mask = (seq_list < batch.seq_len[..., None])
            mask_t = torch.transpose(mask, 0, 1)
            seq_mask = mask_t.unsqueeze(2).repeat(1, 1, cat_x.size(2))

            # Zero out one-hot vectors
            x = cat_x * seq_mask

        logit = self.forward(x, batch.obs[self.ar_input_key])
        logit = logit * beta
        logp = nn.functional.log_softmax(logit, 3)
        gtruth = batch.obs[self.ar_target_key]
        gtruth = gtruth.view(logp.size())
        seq_len = batch.seq_len
        xent, avg_xent, nll_per_card = pred_loss(logp, gtruth, seq_len)

        # v0: [seq, batch, hand_size, bit_per_card]
        v0 = batch.obs["priv_ar_v0"]
        v0 = v0.view(v0.size(0), v0.size(1), self.hand_size, 35)[:, :, :, :25]
        logv0 = v0.clamp(min=1e-6).log()
        _, avg_xent_v0, _ = pred_loss(logv0, gtruth, seq_len)

        return xent, avg_xent, avg_xent_v0, nll_per_card

    def loss_no_grad(self, batch, beta=1, convention_index_override=None):
        with torch.no_grad():
            result =  self.loss(batch, beta=beta, 
                    convention_index_override=convention_index_override)
        return result

    def loss_response_playable(self, batch, beta=1, convention_index_override=None):
        torch.set_printoptions(linewidth=300, precision=2, sci_mode=False)
        x = batch.obs[self.input_key]

        #Append convention one-hot vectors if model is parameterized
        if self.parameterized:
            # Concatenate convention one-hot vectors to x
            convention_indexes = batch.obs[self.convention_idx_key].clone()
            if convention_index_override is not None:
                convention_indexes[:,:] = convention_index_override
            one_hot = F.one_hot(convention_indexes, num_classes=self.num_conventions)
            cat_x = torch.cat((x, one_hot), 2)

            # Create mask to zero out one-hot vectors after sequences finish
            seq_list = torch.arange(cat_x.size(0)).to(batch.seq_len.get_device())
            mask = (seq_list < batch.seq_len[..., None])
            mask_t = torch.transpose(mask, 0, 1)
            seq_mask = mask_t.unsqueeze(2).repeat(1, 1, cat_x.size(2))

            # Zero out one-hot vectors
            x = cat_x * seq_mask

        with torch.no_grad():
            logit = self.forward(x, batch.obs[self.ar_input_key])

        probs = F.softmax(logit, 3)

        should_be_playable = batch.obs["response_should_be_playable"]
        card_position = batch.obs["response_card_position"]
        playable = batch.obs["playable_cards"]

        should_shape = should_be_playable.shape
        should_expanded = should_be_playable[:, :, None, None].expand(probs.shape)
        cards_in_hand = 5
        position_one_hot = F.one_hot(card_position, num_classes=cards_in_hand)
        position_one_hot = position_one_hot[:, :, :, None].expand(probs.shape)
        playable_repeated = torch.unsqueeze(playable, dim=2).repeat(1,1,cards_in_hand,1)
        mask = should_expanded * position_one_hot * playable_repeated
        prob_masked = probs * mask
        probs_sum = torch.sum(prob_masked, axis=3)
        probs_mean = torch.mean(probs_sum[probs_sum != 0])

        return probs_mean.cpu()

    @torch.jit.script_method
    def observe(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsize, num_lstm_layer, num_player, dim = obs["h0"].size()
        h0 = obs["h0"].transpose(0, 1).flatten(1, 2).contiguous()
        c0 = obs["c0"].transpose(0, 1).flatten(1, 2).contiguous()

        s = obs[self.input_key].unsqueeze(0)
        x = self.net(s)
        if self.fc_only:
            o, (h, c) = x, (h0, c0)
        else:
            o, (h, c) = self.lstm(x, (h0, c0))

        h = h.view(num_lstm_layer, bsize, num_player, dim)
        c = c.view(num_lstm_layer, bsize, num_player, dim)
        return {
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }

    @torch.jit.script_method
    def sample(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsize, num_lstm_layer, num_player, dim = obs["h0"].size()
        h0 = obs["h0"].transpose(0, 1).flatten(1, 2).contiguous()
        c0 = obs["c0"].transpose(0, 1).flatten(1, 2).contiguous()

        s = obs[self.input_key].unsqueeze(0)

        if self.parameterized:
            convention_idx = obs[self.convention_idx_key].unsqueeze(0)
            # tensor_convention_index = torch.tensor(convention_idx)
            one_hot = F.one_hot(convention_idx, 
                    num_classes=self.num_conventions)
            s = torch.cat((s, one_hot), 2)

        x = self.net(s)

        if self.fc_only:
            o, (h, c) = x, (h0, c0)
        else:
            o, (h, c) = self.lstm(x, (h0, c0))
        # o: [seq_len(1), batch, dim]
        seq, bsize, hid_dim = o.size()

        assert seq == 1, "seqlen should be 1"
        # assert bsize == 1, "batchsize for BeliefModel.sample should be 1"
        o = o.view(bsize, hid_dim)
        o = o.unsqueeze(1).expand(bsize, self.num_sample, hid_dim)

        in_t = torch.zeros(bsize, self.num_sample, hid_dim // 8, device=o.device)
        shape = (1, bsize * self.num_sample, self.hid_dim)
        ar_hid = (
            torch.zeros(*shape, device=o.device),
            torch.zeros(*shape, device=o.device),
        )
        sample_list = []
        for i in range(self.hand_size):
            ar_in = torch.cat([in_t, o], 2).view(bsize * self.num_sample, 1, -1)

            ar_out, ar_hid = self.auto_regress(ar_in, ar_hid)
            logit = self.fc(ar_out.squeeze(1))
            prob = nn.functional.softmax(logit, 1)

            sample_t = prob.multinomial(1)
            sample_t = sample_t.view(bsize, self.num_sample)
            onehot_sample_t = torch.zeros(
                bsize, self.num_sample, 25, device=sample_t.device
            )
            onehot_sample_t.scatter_(2, sample_t.unsqueeze(2), 1)
            in_t = self.emb(onehot_sample_t)
            sample_list.append(sample_t)

        sample = torch.stack(sample_list, 2)

        h = h.view(num_lstm_layer, bsize, num_player, dim)
        c = c.view(num_lstm_layer, bsize, num_player, dim)
        return {
            "sample": sample,
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }
