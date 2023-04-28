import gym
import string
import time
import argparse
import numpy as np

def main():
    # Commandline arguements:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str,choices=["pong","breakout","enduro","beam_rider","space_invaders","seaquest","qbert"],default="pong")
    parser.add_argument("--image",type=int,choices=[0,1],default=0)
    parser.add_argument("--n_iter",type=int,default=1000)
    args = parser.parse_args()

    # Convert to OpenAI Gym game name
    gamestring = string.capitalize(args.game)
    cu = gamestring.find('_')
    if cu>=0:
        gamestring ="%s%s"%(gamestring[:cu],string.capitalize(gamestring[cu+1:]))
    if not args.image:
        gamestring = "%s-ram-v0"%gamestring
    else:
        gamestring ="%s-v0"%gamestring

    # Make the Atari environment
    env = gym.make(gamestring)
    actions = env.action_space

    # Draw the game:
    env.reset()
    env.render()

    reward_total = 0
    for i in range(args.n_iter):

        # Choose a random action:
        a = actions.sample()

        # Run chosen action. If args.image == 1, observ will be an image.
        # Otherwise it will be the ram state.
        observ, reward, done, step_info = env.step(a)
        reward_total += reward

        # Draw the game:
        env.render()

        if done:
            print("Game Over!")
            break;

        # Pause
        #time.sleep(.15)

    print("Total reward: %d"%reward_total)

if __name__ == "__main__":
    main()
