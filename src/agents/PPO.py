from envs.StockTrading import StockTrading
from stable_baselines3 import PPO
from pandas_datareader import data as pdr
from stable_baselines3.common.vec_env import DummyVecEnv
import time

tickers = ['BNDX','URTH']
start_date = '2019-01-01'
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
data_train = data[:samples_train]
data_test = data[samples_train:]

def train():
    policy = "MlpPolicy"
    train_env = DummyVecEnv([lambda: StockTrading(df = data_train)])
    timesteps = 10000
    runs = 1
    model = PPO(policy,train_env,verbose=0,seed=42)
    start = time.time()
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f'PPO.agent')
    print(f'Train Time : {end-start}')
    return model



