import numpy as np
from util.util import softmax_prob, Message, discount, fmt_row
from util.frozen_lake import rollout
import pdb


def value_iteration(env, gamma, nIt):
    """
    Inputs:
        env: Environment description
        gamma: discount factor
        nIt: number of iterations
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt+1
    """
    # Env[state = S][action = a]  = {[probability that you end up in state s][state s]
    # [reward you get when you get to state s][have you reached the goal or not]}
    Vs = [np.zeros(env.nS)]
    pis = [np.zeros(env.nS, dtype='int')]
    for it in range(nIt):
        V, pi = vstar_backup(Vs[-1], env, gamma)
        Vs.append(V)
        pis.append(pi)
    return Vs, pis


def policy_iteration(env, gamma, nIt):
    """
    Inputs:
        env: Environment description
        gamma: discount factor
        nIt: number of iterations
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt+1
    """
    Vs = [np.zeros(env.nS)]
    pis = [np.zeros(env.nS, dtype='int')]
    for it in range(nIt):
        vpi = policy_evaluation_v(pis[-1], env, gamma)
        qpi = policy_evaluation_q(vpi, env, gamma)
        pi = qpi.argmax(axis=1)
        Vs.append(vpi)
        pis.append(pi)
    return Vs, pis


def policy_gradient_optimize(env, policy, gamma,
                             max_pathlength, timesteps_per_batch, n_iter, stepsize):
    from collections import defaultdict
    stat2timeseries = defaultdict(list)
    widths = (17, 10, 10, 10, 10)
    print fmt_row(widths, ["EpRewMean", "EpLenMean", "Perplexity", "KLOldNew"])
    for i in xrange(n_iter):
        # collect rollouts
        total_ts = 0
        paths = []
        while True:
            path = rollout(env, policy, max_pathlength)
            paths.append(path)
            total_ts += path["rewards"].shape[0]  # Number of timesteps in the path
            # pathlength(path)
            if total_ts > timesteps_per_batch:
                break

        # get observations:
        obs_no = np.concatenate([path["observations"] for path in paths])
        # Update policy
        policy_gradient_step(policy, paths, gamma, stepsize)

        # Compute performance statistics
        pdists = np.concatenate([path["pdists"] for path in paths])
        kl = policy.compute_kl(pdists, policy.compute_pdists(obs_no)).mean()
        perplexity = np.exp(policy.compute_entropy(pdists).mean())

        stats = {"EpRewMean": np.mean([path["rewards"].sum() for path in paths]),
                 "EpRewSEM": np.std([path["rewards"].sum() for path in paths]) / np.sqrt(len(paths)),
                 "EpLenMean": np.mean([path["rewards"].shape[0] for path in paths]),  # pathlength(path)
                 "Perplexity": perplexity,
                 "KLOldNew": kl}
        print fmt_row(widths,
                      ['%.3f+-%.3f' % (stats["EpRewMean"], stats['EpRewSEM']), stats['EpLenMean'], stats['Perplexity'],
                       stats['KLOldNew']])

        for (name, val) in stats.items():
            stat2timeseries[name].append(val)
    return stat2timeseries


#####################################################
## TODO: You need to implement all functions below ##
#####################################################
def vstar_backup(v_n, env, gamma):
    """
    Apply Bellman backup operator V -> T[V], i.e., perform one step of value iteration

    :param v_n: the state-value function (1D array) for the previous iteration
    :param env: environment description providing the transition and reward functions
    :param gamma: the discount factor (scalar)
    :return: a pair (v_p, a_p), where
    :  v_p is the updated state-value function and should be a 1D array (S -> R),
    :  a_p is the updated (deterministic) policy, which should also be a 1D array (S -> A)
    """
    v_p = np.zeros((env.nS,))
    a_p = np.zeros((env.nS,))
    for state, prev_val in enumerate(v_n):

        Qs = np.zeros(len(env.P[state]))
        for action, possible_resulting in env.P[state].iteritems():

            Q = 0
            for prob, next_state, reward, reached_goal in possible_resulting:

                Q += prob * (reward + gamma * v_n[next_state])

            Qs[action] = Q

        a_p[state] = np.argmax(Qs)
        v_p[state] = np.max(Qs)

    assert v_p.shape == (env.nS,)
    assert a_p.shape == (env.nS,)
    return (v_p, a_p)


def policy_evaluation_v(pi, env, gamma):
    """
    :param pi: a deterministic policy (1D array: S -> A)
    :param env: environment description providing the transition and reward functions
    :param gamma: the discount factor (scalar)
    :return: vpi, the state-value function for the policy pi

    Hint: use np.linalg.solve
    """
    Ppi = np.zeros((env.nS, env.nS))
    Rpi = np.zeros((env.nS,))
    for state in xrange(env.nS):

        action = pi[state]

        Q = 0
        for prob, next_state, reward, reached_goal in env.P[state][action]:

            Ppi[state, next_state] += prob

            Q += prob * reward

        Rpi[state] = Q

    a = np.eye(env.nS) - gamma * Ppi
    b = Rpi
    vpi = np.linalg.solve(a, b)

    assert vpi.shape == (env.nS,)
    return vpi


def policy_evaluation_q(vpi, env, gamma):
    """
    :param vpi: the state-value function for the policy pi
    :param env: environment description providing the transition and reward functions
    :param gamma: the discount factor (scalar)
    :return: qpi, the state-action-value function for the policy pi
    """
    qpi = np.zeros((env.nS, env.nA))
    for state, prev_val in enumerate(vpi):

        for action, possible_resulting in env.P[state].iteritems():

            Q = 0
            for prob, next_state, reward, reached_goal in possible_resulting:

                Q += prob * (reward + gamma * vpi[next_state])

            qpi[state, action] = Q

    assert qpi.shape == (env.nS, env.nA)
    return qpi


def softmax_policy_gradient(f_sa, s_n, a_n, adv_n):
    """
    Compute policy gradient of policy for discrete MDP, where probabilities
    are obtained by exponentiating f_sa and normalizing.

    See softmax_prob and softmax_policy_checkfunc functions in util. This function
    should compute the gradient of softmax_policy_checkfunc.

    INPUT:
      f_sa : a matrix representing the policy parameters, whose first dimension s
             indexes over states, and whose second dimension a indexes over actions
      s_n : states (vector of int)
      a_n : actions (vector of int)
      adv_n : discounted long-term returns (vector of float)
    """
    # see slide 9 of lecture 18

    nb_timesteps = len(adv_n)  # =T

    softmaxed_af = softmax_prob(f_sa)

    # initialize the gradient
    grad_sa = np.zeros_like(f_sa)

    for timestep in xrange(nb_timesteps):

        # gamma is in adv_n already
        for action in xrange(f_sa.shape[1]):

            if action == a_n[timestep]:
                grad_sa[s_n[timestep], action] += adv_n[timestep] * (1 - softmaxed_af[s_n[timestep], action])

            else:
                grad_sa[s_n[timestep], action] += adv_n[timestep] * (-softmaxed_af[s_n[timestep], action])

    grad_sa /= nb_timesteps  # not in slides, but have to do; see util.policy_gradient_checkfunc

    # assert grad_sa == (env.nS, env.nA)
    return grad_sa


def policy_gradient_step(policy, paths, gamma, stepsize):
    """
    Compute the discounted returns, compute the policy gradient (using softmax_policy_gradient above),
    and update the policy parameters policy.f_sa
    """
    theta = policy.f_sa
    grad = np.zeros_like(theta)
    nb_runs = len(paths)

    # a bunch of sample runs were made in paths
    for path in paths:

        states_observed = path['observations']
        nb_states_observed = len(states_observed)
        actions_taken = path['actions']
        rewards = path['rewards']

        G_vector = np.zeros((nb_states_observed,))
        prev_reward = 0
        # fill in the G_vector starting from the last observation
        for timestep in xrange(nb_states_observed-1, -1, -1):

            new_reward = np.power(gamma, timestep) * rewards[timestep]
            G_vector[timestep] = new_reward + prev_reward
            prev_reward += new_reward

        # find gradient for this run
        grad += softmax_policy_gradient(theta, states_observed, actions_taken, G_vector)

    # approximate expected value by averaging gradient over all runs
    grad /= nb_runs

    # update theta by talking a step along the gradient (ascent)
    policy.f_sa += stepsize * grad


