import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=0,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.transfrom_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        T = self.activation_function(self.transfrom_layer(x))
        G = self.gate_activation(self.gate_layer(x))

        return T.mul(G) + x.mul(1-G)


class Attention(nn.Module):

    def __init__(self, config):

        super(Attention, self).__init__()

        # input_size = 6d, output_size = 1
        input_size = 6 * (config.hidden_size)
        self.alpha = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(p=(1-config.input_keep_prob))
        self.config = config
        self.JX = 1
        self.M = 1
        self.JQ = 1

    def forward(self, h, u, h_mask=None, u_mask=None):
        
        self.JX = h.size()[2]
        self.M = h.size()[1]
        self.JQ = u.size()[1]

        if self.config.q2c_att or self.config.c2q_att:
            u_a, h_a = self.bi_attention(h, u, h_mask, u_mask)
#            print("u_a:", u_a.size())
#            print("h_a:", h_a.size())
        
        # only implement the simple beta function in the paper
        return torch.cat((h, u_a, h.mul(u_a), h.mul(h_a)), -1) 

    def bi_attention(self, h, u, h_mask=None, u_mask=None):
        
        h_aug = h.unsqueeze(3).repeat(1, 1, 1, self.JQ, 1)
        u_aug = u.unsqueeze(1).unsqueeze(1).repeat(1, self.M, self.JX, 1, 1)

        h_u = h_aug.mul(u_aug)

        # similariy matrix S
        s = self.dropout(torch.cat((h_aug, u_aug, h_u), -1))
        
        s = self.alpha(s).squeeze(-1)

        # compute context to query
        u_a = softsel(u_aug, s, 3)
        h_a = softsel(h, s.max(dim=3)[0], 2)
        h_a = h_a.unsqueeze(2).repeat(1, 1, self.JX, 1)
        return u_a, h_a
    
def softsel(target, logits, dim=0):

    """
    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :return: [..., d], dtype=float
    """
    a = F.softmax(logits, dim=dim)
    target_rank = len(list(target.size()))
    dim_to_reduce = target_rank - 2
    out = target.mul(a.unsqueeze(-1)).sum(dim_to_reduce).squeeze(dim_to_reduce)

    return out
