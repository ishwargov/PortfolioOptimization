from tensorflow.python.ops.math_ops import Prod
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class TransformerModel(TorchModelV2,nn.Module):

    def __init__(self,  obs_space, action_space, num_outputs, model_config, name):
        super().__init__( obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # Define the transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=406, nhead=7, batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.action_shape = action_space.shape
        self.obs_shape = obs_space.shape
        self.prod = reduce(lambda a, b: a * b, self.obs_shape)
        # Define the policy head.
        self.policy_head = torch.nn.Linear(self.prod, 2*self.action_shape[0])
        self.critic_head = torch.nn.Linear(self.prod, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        batch,time,stock,features = obs.shape
        obs_t = obs.view(batch,time,stock*features)
        network_output = self.encoder(obs_t)
        network_output = network_output.view(batch,-1)
        value = self.critic_head(network_output)
        self._value = value.reshape(-1)
        logits = self.policy_head(network_output)
        return logits, state

    def value_function(self):
        return self._value


config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(
        env = PortfolioEnv,
        env_config={
        'stock_data' : x_train,
        'window_size' : 7
        },
      )
    .rollouts(num_rollout_workers=1)
    .training(
            model={
                "custom_model": TransformerModel,
                "vf_share_layers": True,
            }
        )
    .evaluation(evaluation_num_workers=1)
    .framework('torch')
)

algo = config.build()  # 2. build the algorithm,

for _ in range(3):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.
