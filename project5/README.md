
## Requirements for 1_MDP

 + NOTE: Requires Scipy 0.15.0 or more recent and OpenAI Gym: https://gym.openai.com/docs
 + pip install numpy
 + pip install --upgrade scipy
 + pip install gym==0.8.0
 + pip install numdifftools

## Requirements for 2_PGO

 + tensorflow: https://www.tensorflow.org/install/
 
## Requirements for 3_ATARI

 1. Install Atari dependancies: pip install gym[atari]
    + This allows access to the games here: https://gym.openai.com/envs#atari
 2. Install h5py
    + sudo apt-get install libhdf5-dev
    + pip uninstall h5py
    + pip install --no-cache-dir h5py
 3. Install cgt
    + Get the git repo: https://github.com/joschu/cgt
    + export PYTHONPATH=/path/to/cgt

 4. OPTION A: Run the Proximal Policy Optimization implementation (A variant of [Trust Region Policy Optimization](http://arxiv.org/abs/1502.05477) ) by running inside atari_extra:

    python learn_atari_gym.py --outfile <choose name for output> --metadata <choose file name to output metadata to>
    
    There are many other parameters which can be set when calling this function. Here are a few interesting ones:
    -- game (choose between pong, breakout, enduro, beam_rider, space_invaders, seaquest, qbert, default=pong)
    -- plot (show game, default=0)

 5. OPTION B: Instead of the provided ppo.py you can try to run this implementation: https://github.com/joschu/modular_rl

## Atari Games

 + beam_rider
 + breakout
 + enduro
 + pong
 + qbert
 + space_invaders

## BeamRider has 9 actions:

* 0: Do Nothing
* 1: Fire
* 2: Up
* 3: Right
* 4: Left
* 5: Up & Right
* 6: Up & Left
* 7: Right & Fire
* 8: Left & Fire

