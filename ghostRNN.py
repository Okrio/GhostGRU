
import torch
import torch.nn as nn
import numpy as np
import math



class GhostGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, ghost_ratio, has_bias=None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ghost_ratio = ghost_ratio
        self.has_bias = has_bias
        self.ghost_size = hidden_size - (hidden_size // ghost_ratio)
        self.weight_gate = nn.Parameter(
            torch.Tensor(np.random.randn(2 * (hidden_size // ghost_ratio), input_size + hidden_size).astype(np.float32))
        )
        self.weight_candidate = nn.Parameter(
            torch.Tensor(np.random.randn(1 * (hidden_size // ghost_ratio), input_size + hidden_size).astype(np.float32))
        )
        self.weight_ghost = nn.Parameter(
            torch.Tensor(np.random.randn(self.ghost_size, (hidden_size // ghost_ratio)).astype(np.float32))
        )

        if self.has_bias:
            self.bias_gate = nn.Parameter(
                torch.Tensor(np.random.randn(2 * (hidden_size//ghost_ratio)).astype(np.float32))
            )
            self.bias_candidate = nn.Parameter(
                torch.Tensor(np.random.randn(1 * (hidden_size // ghost_ratio)).astype(np.float32))
            )
            self.bias_ghost = torch.Tensor(np.random.randn(self.ghost_size).astype(np.float32))
        else:
            self.bias_gate = None
            self.bias_candidate = None
            self.bias_ghost = torch.zeros(self.ghost_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def one_run(self, inputs, hidden):
        '''
        inputs: b F
        hidden: b F
        ouptut:b F
        '''
        gate_inputs = torch.cat([inputs, hidden], 1)
        gate_inputs = torch.matmul(gate_inputs,  self.weight_gate.transpose(1,0))

        if self.bias_gate is not None:
            gate_inputs = gate_inputs + self.bias_gate

        value = torch.nn.Sigmoid()(gate_inputs)
        r, u = torch.split(value, value.shape[-1]//2 ,1)
        r_state = r * hidden[:, self.weight_ghost.shape[0]:]

        candidate = torch.cat([inputs, hidden[:,:self.weight_ghost.shape[0]], r_state], 1) 
        candidate = torch.matmul(candidate, self.weight_candidate.transpose(1,0))
        if self.bias_gate is not None:
            candidate = candidate + self.bias_candidate
        c = torch.nn.Tanh()(candidate)
        new_h = u * hidden[:, self.weight_ghost.shape[0]:] + (1 - u)*c
        ghost_state = torch.matmul(new_h , self.weight_ghost.transpose(1,0)) + self.bias_ghost
        ghost_state = torch.nn.Tanh()(ghost_state)

        new_h_with_ghost = torch.cat([ghost_state, new_h],1)
        return new_h_with_ghost,new_h_with_ghost
    def forward(self, x, hidden=None):
        if x.ndim>2:
            b,t,f = x.shape
        else:
            x = x.unsqueeze(1)
            b,t,f = x.shape
            # t = 1
        hidden = hidden.squeeze(1)
        outputs = []
        for i in range(t):
            x_t = x[:,i,:]
            hidden,_ = self.one_run(x_t, hidden)
            outputs.append(hidden)
        outputs = torch.stack(outputs,1)
        hidden = hidden.unsqueeze(1)
        return outputs, hidden


def test_ghostGRUCell():
    inp = torch.randn(1,2,8)
    hidden = torch.randn(1,1,8)
    
    net = GhostGRUCell(input_size=8,hidden_size=8,ghost_ratio=3, has_bias=True).eval()
    out = net(inp, hidden)

    print('sc')


if __name__ == "__main__":
    test_ghostGRUCell()
