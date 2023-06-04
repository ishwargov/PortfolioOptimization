from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray import air
from ray import tune
config = DDPGConfig()
# Print out some default values.
print(config.lr)
# Update the config object.
config = config.training(lr=tune.grid_search([0.001, 0.0001]))
# Set the config object's env.
config = config.environment(env="Pendulum-v1")
# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(
    "DDPG",
    run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
    param_space=config.to_dict(),
).fit()
