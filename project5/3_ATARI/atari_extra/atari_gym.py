from rl_gym import MDP, Serializable
import numpy as np
import sys
import gym
#sys.path.append("vendor/Arcade-Learning-Environment")
#from ale_python_interface import ALEInterface #pylint: disable=F0401

OBS_RAM = 0
OBS_IMAGE = 1



def to_rgb(ale):
    (screen_width,screen_height) = ale.getScreenDims()
    arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    ale.getScreenRGB(arr)
    # The returned values are in 32-bit chunks. How to unpack them into
    # 8-bit values depend on the endianness of the system
    if sys.byteorder == 'little': # the layout is BGRA
        arr = arr[:,:,0:3].copy() # (0, 1, 2) <- (2, 1, 0)
    else: # the layout is ARGB (I actually did not test this.
          # Need to verify on a big-endian machine)
        arr = arr[:,:,2:-1:-1]
    img = arr
    return img

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class AtariMDP(MDP, Serializable):

    def __init__(self, rom_name, frame_skip=4):

        #self.ale = ALEInterface()
        #self.ale.loadROM(rom_path)
        self.env = gym.make(rom_name)

        # Things returned at each step.
        self.observation = self.env.reset();
        self.done = False

        self._rom_name = rom_name

        if rom_name.find("-ram-") > -1:
            self._obs_type = OBS_RAM
        else:
            self._obs_type = OBS_IMAGE
        Serializable.__init__(self, rom_name, self._obs_type, frame_skip)
        self.options = (rom_name, self._obs_type, frame_skip)

        if type(self.env.action_space) == type(gym.spaces.discrete.Discrete(1)):
            self._action_space = np.arange(self.env.action_space.n)
        # ELSE TODO

        self.frame_skip = frame_skip


    def get_image(self):
        if self._obs_type == OBS_IMAGE:
            return self.observation
        else:
            return 0
    def get_ram(self):
        if self._obs_type == OBS_RAM:
            return self.observation
        else:
            return 0
    def game_over(self):
        return self.done
    def reset_game(self):
        self.observation = self.env.reset()
        self.done = False
        return self.observation

    @property
    def n_actions(self):
        return len(self._action_space)

    def get_obs(self):
        if self._obs_type == OBS_RAM:
            return self.get_ram()[None,:]
        else:
            assert self._obs_type == OBS_IMAGE
            return self.get_image()[None,:,:,:]

    def step(self, a):

        total_reward = 0.0
        action = self.action_space[a]
        for _ in xrange(self.frame_skip):
            observation,reward,done,info = self.env.step(action)
            total_reward += reward
            self.done = done
            self.observation = observation
        ob = self.get_obs().reshape(1,-1)
        return ob, np.array([total_reward]), self.done

    # return: (states, observations)
    def reset(self):
        self.reset_game()
        return self.get_obs()

    @property
    def action_space(self):
        return self._action_space

    def plot(self):
        self.env.render()
