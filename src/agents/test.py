
import time
import sys
import gym
from stable_baselines3.common.env_checker import check_env
from pandas_datareader import data as pdr
if True:
    sys.path.append("..")
    from envs.StockTrading import StockEnv


tickers = ['BNDX', 'URTH']
start_date = '2012-01-01'
end_date = "2022-11-20"
df = pdr.get_data_yahoo([tickers][0], start=start_date, end=end_date)
print(df.shape)
# remove NaN values
data = df.copy()
data['Adj Close'] = data['Adj Close'].ffill()
df = df.bfill(axis=1)
data['Adj Close'] = data['Adj Close'].bfill()
df = df.bfill(axis=1)
data = data['Adj Close']

train_pct = 0.8
samples_train = int(train_pct*len(data))
#print(len(data))
data_train = data[:samples_train]
data_test = data[samples_train:]

max_trade = 30
balance = 10000
transaction_fee = 0.001

env = StockEnv(data)
#check_env(env)
t = 0
while True:
   t += 1
   env.render()
   observation = env.reset()
   print(observation)
   action = env.action_space.sample()
   observation, reward, done, info = env.step(action)
   if done:
    print(f"Episode finished after {t} timesteps")
   break
env.close()