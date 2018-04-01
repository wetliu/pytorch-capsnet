import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import layers
from torch.autograd import Variable

class PrimaryCap(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_routing, input_atoms,
                 output_atoms, stride, kernel_size, padding, leaky):
        super(PrimaryCap, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.leaky = leaky
        self.input_atoms = input_atoms

        self.conv = nn.Conv2d(in_channels=input_atoms,
            out_channels=output_dim * (output_atoms),
            kernel_size=kernel_size,
            stride=stride,
            bias=False)
        nn.init.normal(self.conv.weight.data, mean=0,std=0.1)

        self.bias = nn.Parameter(torch.Tensor(output_dim, output_atoms, 1, 1))
        nn.init.constant(self.bias.data, val=0.1)

    def forward(self, x):
        x = x.view(x.size(0)*self.input_dim, self.input_atoms, x.size(3), x.size(4))
        x = self.conv(x)

        votes = x.view(x.size(0), self.input_dim, self.output_dim, self.output_atoms, x.size(2), x.size(3))

        tile_shape = list(self.bias.size())
        tile_shape[2], tile_shape[3] = x.size(2), x.size(3)

        biases_replicated = self.bias.expand(tile_shape)
        logit_shape = [x.size(0),self.input_dim,self.output_dim,x.size(-2),x.size(-1)]

        return _routing(votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_routing=1,
            is_digit_cap=False,
            leaky=False)

class DigitCap(torch.nn.Module):
    def __init__(self, input_dim, output_dim, input_atoms,# + coord_atom,
                 output_atoms, num_routing, leaky):
        super(DigitCap, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.leaky = leaky

        self.weights = nn.Parameter(torch.Tensor(input_dim, input_atoms, output_dim * output_atoms))
        nn.init.normal(self.weights.data, mean=0,std=0.1)

        self.bias = nn.Parameter(torch.Tensor(output_dim, output_atoms))
        nn.init.constant(self.bias.data, val=0.1)

    def forward(self, x):
        tile_shape = list(x.size())
        tile_shape[-1] = self.output_dim * self.output_atoms

        x = x.expand(tile_shape)
        votes = torch.sum(x * self.weights, dim=2)

        votes = votes.view(votes.size(0), self.input_dim, self.output_dim, self.output_atoms)

        logit_shape = [x.size(0),self.input_dim,self.output_dim]

        return _routing(votes=votes,
            biases=self.bias,
            logit_shape=logit_shape,
            num_dims=4,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_routing=3,
            is_digit_cap=True,
            leaky=False)

class Reconstruction(torch.nn.Module):
    def __init__(self, num_classes, num_atoms, layer_sizes, num_pixels):
        super(Reconstruction, self).__init__()

        self.num_atoms = num_atoms

        first_layer_size, second_layer_size = layer_sizes

        """tried to follow the official code, with mean=0, std=0.1 and bias = 0.1
           since pytorch does not have truncated normal initializer yet, I will
           stick with the following setting now."""
        self.dense0 = nn.Linear(num_atoms*num_classes, first_layer_size)
        nn.init.normal(self.dense0.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense0.bias.data, val=0.0)
        self.dense1 = nn.Linear(first_layer_size, second_layer_size)
        nn.init.normal(self.dense1.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense1.bias.data, val=0.0)
        self.dense2 = nn.Linear(second_layer_size, num_pixels)
        nn.init.normal(self.dense2.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense2.bias.data, val=0.0)

    def forward(self, capsule_embedding, capsule_mask): #
        capsule_mask_3d = capsule_mask.clone().unsqueeze_(-1)
        tile_shape = list(capsule_mask_3d.size())
        tile_shape[-1] = self.num_atoms

        atom_mask = capsule_mask_3d.expand(tile_shape)
        filtered_embedding = capsule_embedding * atom_mask
        filtered_embedding = filtered_embedding.view(filtered_embedding.size(0), -1)

        x = F.relu(self.dense0(filtered_embedding))
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x

def _routing(votes,
            biases,
            logit_shape,
            num_dims,
            input_dim,
            output_dim,
            num_routing,
            is_digit_cap,
            leaky=False):
    input_atom = votes.size(3)
    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
        votes_t_shape += [i + 4]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
        r_t_shape += [i + 4]

    votes_trans = votes.permute(votes_t_shape)
    batch_size = votes.size(0)
    votes_trans_stopped = votes_trans.clone().detach()

    logits = Variable(torch.zeros(logit_shape)).cuda()

    for i in range(num_routing):
        route = F.softmax(logits, 2)

        if i == num_routing - 1:
            preactivate_unrolled = route * votes_trans
            preact_trans = preactivate_unrolled.permute(r_t_shape)
            preactivate = torch.sum(preact_trans, dim=1) + biases
            activation = _squash(preactivate)
        else:
            preactivate_unrolled = route * votes_trans_stopped
            preact_trans = preactivate_unrolled.permute(r_t_shape)
            preactivate = torch.sum(preact_trans, dim=1) + biases
            activation = _squash(preactivate)

            act_3d = activation.unsqueeze_(1)
            tile_shape = list(act_3d.size())
            tile_shape[1] = input_dim

            act_replicated = act_3d.expand(tile_shape)
            distances = torch.sum(votes * act_replicated, dim=3)
            logits = logits + distances
    return activation

def _squash(input_tensor):
    norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
