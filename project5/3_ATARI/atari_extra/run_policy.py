import h5py
import cgt
import pickle
import numpy as np
from atari_gym import AtariMDP
from rl_gym import animate_rollout
import time

file = h5py.File('mycrazyoutput','r')
pol = pickle.loads(file['policy_pickle'].value)

mdp = AtariMDP('BeamRider-ram-v0')

for i in range(1000):
  ppo_step = pp.step(mdp.get_obs())
  mdp.step(ppo_step['action'][0])
  mdp.plot()
  time.sleep(.5)
  if mdp.game_over():
	  break


