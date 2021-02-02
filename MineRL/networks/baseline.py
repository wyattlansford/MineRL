import torch.nn as nn
import numpy as np
import torch
import gym

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class CNNLSTM(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        EMBED_SIZE=256
        LSTM_LAYERS = 1
        self.num_layers = LSTM_LAYERS
        self.hidden_size = EMBED_SIZE

        self._cnn = nn.Sequential(
            nn.Conv2d(1, 64, 4, 4), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 4),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 4),
            nn.ReLU(),
        )
        self._inventory = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU()
        )
        
        # self._prev_action_net = nn.Embedding(
        #     action_space.n, EMBED_SIZE
        # )
        # self._prev_value_net = nn.Sequential(
        #     nn.Linear(1, EMBED_SIZE),
        #     nn.Tanh(),
        # )
        self.body = nn.LSTM(512, 256, batch_first=True)

        self._value_head = nn.Linear(256, 1)
        self._policy_head = nn.Linear(256, 256)


    def value_function(self):
        values = self._value_head(self._logits).squeeze()
        return values
        return 

    def forward(self, input_dict, state, seq_lens):

        device = next(self.parameters()).device
        encoding = input_dict['obs'][1].squeeze(0) # Batch*seq, features
        cnn_input = input_dict['obs'][0].permute(0, 3, 1, 2)  # Batch*seq, channel, x, y

        cnn_embedding = self._cnn(cnn_input.to(device))
        cnn_embedding = torch.reshape(cnn_embedding, cnn_embedding.shape[:2])
        encoding_embedding = self._inventory(encoding.to(device))

        # previous action input as discrete
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
        batch_n = body_inputs.size(0) 
        state_size = body_inputs.size(1)
        body_inputs = torch.reshape(body_inputs, (batch_n, batch_t, state_size))
        if not state: # WHY IS INITIAL STATE NOT WORKING? Does this slow down w/ dynamic graph stuff?
            lstm_out, lstm_state = self.body(body_inputs)
        else:

            lstm_out, lstm_state = self.body(body_inputs, state)
        self._logits = torch.reshape(lstm_out, (32, 256))

        outputs = self._policy_head(self._logits)
        lstm_state = [x.permute(1, 0, 2) for x in lstm_state]
        return outputs, lstm_state

    def get_initial_state(self):
        device = self._cnn[0].weight.device
        dtype = self._cnn[0].weight.dtype

        h_c = [
            # shape -> (num_layers * num_directions, num_lstm_nodes)
            torch.zeros((1, 256), dtype=dtype, device=device, requires_grad=False),
            torch.zeros((1, 256), dtype=dtype, device=device, requires_grad=False),
        ]

        return h_c

def register():
    ModelCatalog.register_custom_model('cnn_lstm', CNNLSTM)