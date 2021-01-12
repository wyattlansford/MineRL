import torch.nn as nn
import numpy as np
import torch
import gym

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CNNLSTM(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        EMBED_SIZE=256
        LSTM_LAYERS = 1
        self.num_layers = LSTM_LAYERS
        self.hidden_size = EMBED_SIZE

        self._cnn = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 4),
            nn.ReLU(),
            nn.Conv2d(128, EMBED_SIZE, 4, 4),
            nn.ReLU(),
        )
        self._inventory = nn.Sequential(
            nn.Linear(64, EMBED_SIZE),
            nn.LeakyReLU()
        )
        
        # self._prev_action_net = nn.Embedding(
        #     action_space.n, EMBED_SIZE
        # )
        # self._prev_value_net = nn.Sequential(
        #     nn.Linear(1, EMBED_SIZE),
        #     nn.Tanh(),
        # )
        self.body = nn.LSTM(EMBED_SIZE, EMBED_SIZE, LSTM_LAYERS, **kwargs)

        self._value_head = nn.Sequential(
                nn.Linear(EMBED_SIZE, 1),
            )
        self._policy_head = nn.Sequential(
            nn.Embedding(EMBED_SIZE, action_space.n)
        )


    def value_function(self):
        values = self._value_head(self._logits).squeeze()
        return values
        return 

    def forward(self, input_dict, state, seq_lens):
        device = next(self.parameters()).device
        encoding = input_dict['obs'][1]
        cnn_input = input_dict['obs'][0].permute(0, 3, 1, 2)  # n,c,h,w
        
        cnn_embedding = self._cnn(cnn_input.to(device))
        cnn_embedding = torch.reshape(cnn_embedding, cnn_embedding.shape[:2])
        encoding_embedding = self._inventory(encoding.to(device))

        # previous action input as discrete
        print(input_dict.keys())
        print("input_dict")
        # prev_actions = input_dict['prev_actions']

        # prev_actions = prev_actions.long()
        # action_embedding = self._prev_action_net(prev_actions.to(device))

        # Prev value as discrete
        # prev_rewards = input_dict['prev_rewards']
        # prev_rewards = torch.reshape(prev_rewards, (-1, 1))
        # reward_embedding = self._prev_value_net(prev_rewards.to(device))

        

        # body_inputs = torch.cat((cnn_embedding, encoding_embedding, action_embedding, reward_embedding), dim=-1)
        body_inputs = torch.cat((cnn_embedding, encoding_embedding), dim=-1)


        batch_t = 1
        batch_n = state_inputs.size(0) 
        state_size = state_inputs.size(1)

        state_inputs = torch.reshape(state_inputs, (batch_t, batch_n, state_size))
        lstm_out, lstm_state = self.body(state_inputs, state)

        self._logits = torch.reshape(lstm_out, (batch_t * batch_n, self._rnn.hidden_size))

        outputs = self._policy_head(self._logits)
        return outputs, lstm_state


    def get_initial_state(self):
        h_0 = torch.zeros(self.num_layers, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, self.hidden_size)
        return [h_0, c_0]

    
    def lstm_forward(self, x, state):
        # seq, batch, feature
        if len(state) == 2:
            h_0, c_0 = state
            # layers, batch, feature
            h_0 = h_0.to(self.device).permute(1, 0, 2)
            c_0 = c_0.to(self.device).permute(1, 0, 2)
        elif len(state) == 0:
            h_0, c_0 = None, None
        else:
            raise NotImplementedError
        output, (h_n, c_n) = self.body(x.to(self.device), (h_0, c_0))
        # [tensor(batch, layers, feature)]
        lstm_state = [h_n.permute(1, 0, 2), c_n.permute(1, 0, 2)]
        return output, lstm_state

def register():
    ModelCatalog.register_custom_model('cnn_lstm', CNNLSTM)