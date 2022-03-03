import torch
from typing import Tuple, Dict

class ConventionBelief(torch.jit.ScriptModule):
    def __init__(self, device, hand_size, num_sample):
        super().__init__()
        self.device = device
        self.hand_size = hand_size
        self.num_sample = num_sample

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (1, batchsize, 512)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def sample(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in obs.item():
            print(key)
        bsize, num_lstm_layer, num_player, dim = obs["h0"].size()
        # h0 = obs["h0"].transpose(0, 1).flatten(1, 2).contiguous()
        # c0 = obs["c0"].transpose(0, 1).flatten(1, 2).contiguous()

        # s = obs[self.input_key].unsqueeze(0)
        # x = self.net(s)
        # if self.fc_only:
            # o, (h, c) = x, (h0, c0)
        # else:
            # o, (h, c) = self.lstm(x, (h0, c0))
        # # o: [seq_len(1), batch, dim]
        # seq, bsize, hid_dim = o.size()

        # assert seq == 1, "seqlen should be 1"
        # # assert bsize == 1, "batchsize for BeliefModel.sample should be 1"
        # o = o.view(bsize, hid_dim)
        # o = o.unsqueeze(1).expand(bsize, self.num_sample, hid_dim)

        # in_t = torch.zeros(bsize, self.num_sample, hid_dim // 8, device=o.device)
        # shape = (1, bsize * self.num_sample, self.hid_dim)
        # ar_hid = (
            # torch.zeros(*shape, device=o.device),
            # torch.zeros(*shape, device=o.device),
        # )
        # sample_list = []
        # for i in range(self.hand_size):
            # ar_in = torch.cat([in_t, o], 2).view(bsize * self.num_sample, 1, -1)
            # ar_out, ar_hid = self.auto_regress(ar_in, ar_hid)
            # logit = self.fc(ar_out.squeeze(1))
            # prob = nn.functional.softmax(logit, 1)
            # sample_t = prob.multinomial(1)
            # sample_t = sample_t.view(bsize, self.num_sample)
            # onehot_sample_t = torch.zeros(
                # bsize, self.num_sample, 25, device=sample_t.device
            # )
            # onehot_sample_t.scatter_(2, sample_t.unsqueeze(2), 1)
            # in_t = self.emb(onehot_sample_t)
            # sample_list.append(sample_t)

        # sample = torch.stack(sample_list, 2)

        # h = h.view(num_lstm_layer, bsize, num_player, dim)
        # c = c.view(num_lstm_layer, bsize, num_player, dim)
        sample = torch.zeros((bsize, self.num_sample, self.hand_size), 
                dtype=torch.long)
        return {
            "sample": sample,
            "h0": obs["h0"],
            "c0": obs["c0"],
        }
        
