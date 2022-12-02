from stable_baselines3 import PPO
from pandas_datareader import data as pdr
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import sys
import seaborn as sns

if True:
    sys.path.append("..")
    from envs.StockTrading import StockEnv

sns.set_theme()

tickers = ['URTH', 'BNDX']
start_date = '2012-01-01'
end_date = "2022-11-20"
df = pdr.get_data_yahoo([tickers][0], start=start_date, end=end_date)
# remove NaN values
data = df.copy()
data['Adj Close'] = data['Adj Close'].ffill()
df = df.bfill(axis=1)
data['Adj Close'] = data['Adj Close'].bfill()
df = df.bfill(axis=1)
data = data['Adj Close']

train_pct = 0.8
samples_train = int(train_pct*len(data))
# print(len(data))
data_train = data[:samples_train]
data_test = data[samples_train:]

plt.rcParams["figure.figsize"] = (10, 6)
for i in tickers:
    plt.plot(data[i], label=i)

# only one line may be specified; full height
plt.legend()
plt.margins(x=0)
plt.ylabel("Closing Share Prise (USD)")
plt.xlabel("Time (Years)")
plt.savefig('StockPrice.png', bbox_inches='tight')

# start training and predicting
runs = 5
timesteps = 4000

max_trade = 10
balance = 1000
transaction_fee = 0.001


def train():
    max_trade = 10
    balance = 1000
    transaction_fee = 0.001
    policy = "MlpPolicy"
    train_env = DummyVecEnv([lambda: StockEnv(df=data_train)])
    model = PPO(policy, train_env, verbose=1)
    start = time.time()
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f'PPO.agent')
    print(f'Train Time : {end-start}')
    return model


def predict():
    max_trade = 10
    balance = 1000
    transaction_fee = 0.001
    actions_memory = []
    env = DummyVecEnv([lambda: StockEnv(df=data_test)])
    obs = env.reset()
    for i in range(len(data_test.index.unique())):
        action, _states = ppo.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i == (len(data_test.index.unique()) - 2):
            actions_memory = env.env_method(method_name="get_action_memory")
    return actions_memory[0]


length = data_test.shape[0]
stocks = data_test.shape[1]
Cumulative_returns_daily_drl_ppo = np.zeros([runs, length])
portfolio_weights_ppo = np.zeros([runs, length, stocks])

i = 0
while (i < runs):
    ppo = train()
    portfolio_weights_ppo[i] = np.array(predict())
    return_stocks = data_test.pct_change()
    return_stocks_ppo = np.sum(return_stocks.multiply(
        portfolio_weights_ppo[i]), axis=1)
    print(portfolio_weights_ppo[i])
    Cumulative_returns_daily_drl_ppo[i] = (1+return_stocks_ppo).cumprod()
    i = i+1


# plotting the results
def plot(weights, cumulative):
    fig, axs = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)

    ppo_mean = np.mean(np.array(cumulative), axis=0)
    ppo_std = np.std(np.array(cumulative), axis=0)

    axs[0, 0].plot(data_test.index, ppo_mean, color='tab:blue', linewidth=2.0)
    axs[0, 0].fill_between(data_test.index, ppo_mean - ppo_std,
                           ppo_mean + ppo_std, alpha=0.2, color='tab:blue')
    axs[0, 0].margins(x=0)
    axs[0, 0].margins(y=0)
    axs[0, 0].set_ylim(0)
    axs[0, 0].axhline(1, color='black', linestyle='--', lw=2)
    axs[0, 0].set_ylabel("Cumulative Returns")
    axs[0, 0].set_xlabel("Time (Years-Months)")

    weights_ = np.mean(np.array(weights), axis=0)
    df = pd.DataFrame(weights_,
                      index=data_test.index, columns=tickers)
    stock_dat = ()
    for t in tickers:
        stock_dat += (df[t],)
    axs[0, 1].stackplot(data_test.index, stock_dat, labels=tickers)
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].margins(x=0)
    axs[0, 1].margins(y=0)
    axs[0, 1].set_ylabel("Weights (%)")
    axs[0, 1].set_xlabel("Time (Years-Months)")

    weights_ = weights[np.argmax(
        cumulative[:, -1])]
    df = pd.DataFrame(weights_,
                      index=data_test.index, columns=tickers)
    stock_dat = ()
    for t in tickers:
        stock_dat += (df[t],)
    axs[1, 0].stackplot(data_test.index, stock_dat, labels=tickers)
    axs[1, 0].margins(x=0)
    axs[1, 0].margins(y=0)
    axs[1, 0].set_ylabel("Weights (%)")
    axs[1, 0].set_xlabel("Time (Years-Months)")

    weights_ = weights[np.argmin(
        cumulative[:, -1])]
    df = pd.DataFrame(weights_,
                      index=data_test.index, columns=tickers)
    stock_dat = ()
    for t in tickers:
        stock_dat += (df[t],)
    axs[1, 1].stackplot(data_test.index, stock_dat, labels=tickers)
    axs[1, 1].margins(x=0)
    axs[1, 1].margins(y=0)
    axs[1, 1].set_ylabel("Weights (%)")
    axs[1, 1].set_xlabel("Time (Years-Months)")

    plt.savefig('ppo.png', bbox_inches='tight')


plot(portfolio_weights_ppo, Cumulative_returns_daily_drl_ppo)
