import numpy as np


def discount(x: np.ndarray, discount_factor=0.95, bootstrap_value=0.0):
    """
        Computes a discounted array from end (t == T) to start (t == 0):
            discounted[T + 1] = bootstrap_value
            discounted[t] = x[t] + discount_factor * discounted[t + 1]
    """
    discounted = np.zeros_like(x)
    next_discounted_val = bootstrap_value
    i = 0
    for val in x[::-1]:
        next_discounted_val = val + discount_factor * next_discounted_val
        discounted[i] = next_discounted_val
        i += 1
    return discounted[::-1]


def gae_delta(rewards: np.ndarray, values: np.ndarray, discount_factor=0.95, bootstrap_value=0.0):
    """
        Generalized Advantage Estimation Delta:
            Computed from end (t == T) to start (t == 0)
            delta[T] = r[t] + discount_factor * bootstrap_value - V[T]
            delta[t] = r[t] + discount_factor * V[t + 1] - V[t]

        Please see https://arxiv.org/pdf/1506.02438.pdf for more details
    """
    delta = np.zeros_like(values)
    delta[-1] = rewards[-1] + discount_factor * bootstrap_value - values[-1]
    for t in reversed(range(len(rewards) - 1)):
        # Is state is terminal we don't expect anything
        if values[t] == 0.0:
            delta[t] = 0.0
        else:
            delta[t] = rewards[t] + discount_factor * values[t + 1] - values[t]
    return delta


def general_advantage_estimation(rewards: np.ndarray, values: np.ndarray, discount_factor=0.95, lam=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float)
    delta = gae_delta(rewards, values, discount_factor)
    for t in range(len(rewards)):
        for l in range(len(rewards) - t - 1):
            advantages[t] += np.power(discount_factor * lam, l) * delta[t + l]
    return advantages


def generalized_advantage_estimation(rewards: np.ndarray,
                                     values: np.ndarray,
                                     discount_factor=0.95,
                                     lam=0.9,
                                     bootstrap_value=0.):
    """
        Generalized Advantage Estimation:
            A^GAE(gamma, lambda) [t] = sum l in [0, +inf)  pow(gamma * lambda, l) * delta[t + l]

        Please see https://arxiv.org/pdf/1506.02438.pdf for more details
    """
    return discount(gae_delta(rewards, values, discount_factor), discount_factor * lam, bootstrap_value=bootstrap_value)


def expected_return(rewards: np.ndarray, values: np.ndarray, discount_factor=0.95, bootstrap_value=0.):
    """
        Return expectations:
            Computed from end (t == T) to start (t == 0)
            R[T] = r[T] + discount_factor * bootstrap_value
            R[t] = r[t] + discount_factor * R[t + 1]
    """
    returns = np.zeros_like(values)
    returns[-1] = rewards[-1] + discount_factor * bootstrap_value
    for i in reversed(range(len(rewards) - 1)):
        # Is state is terminal we don't expect anything
        if values[i] == 0.0:
            returns[i] = 0.0
        else:
            returns[i] = rewards[i] + discount_factor * returns[i + 1]
    return returns


def expectations(rewards: np.ndarray,
                 values: np.ndarray,
                 next_is_terminal: np.ndarray,
                 discount_factor=0.95,
                 lam=0.95,
                 bootstrap_value=0.0):
    """
    Computes expected returns and advantages given rewards and values

    Preliminaries:
        V[t] - values
        r[t] - rewards
        T - time horizon, so t in range [0, T]
        V[t == T + 1] = bootstrapped_value
        V of terminal state is 0.0

    Returns:
        R[t == T] = r[t] + gamma * V[t + 1]
        R[t] = r[t] + gamma * R[t + 1]

        if (t + 1) is a terminal state:
            R[t] = r[t]   since V[t + 1] == 0.0  (value of a terminal state)

    Delta:
        delta[t] = r[t] + gamma * V[t + 1] - V[t]

        if (t + 1) is a terminal state:
            delta[t] = r[t] - V[t]   since V[t + 1] == 0.0  (value of a terminal state)

    Advantage:
        Generalized Advantage Estimation:
            A^GAE(gamma, lambda) [t] = sum l in [0, +inf)  pow(gamma * lambda, l) * delta[t + l]
            source: https://arxiv.org/pdf/1506.02438.pdf

        A[t == T] = delta[t]  assuming that A[t == t + 1] == 0.0
        A[t] = delta[t] + gamma * lam * A[t + 1]

    :param rewards: r[t] - Array of the actual rewards for state and taken action in rollout
    :param values: V[t] - Array of values predicted by the model for state in rollout
    :param next_is_terminal: Array of bool values indicating whether action leads to the terminal state
    :param discount_factor: gamma [0, 1] - how much we aware of future rewards
    :param lam: lambda [0, 1] - bias/variance controller, Lambda == 0 -  Lambda=1
    :param bootstrap_value: V[T + 1] The value of the next state in rollout
    :return: Array of expected returns, Array of advantages
    """

    #
    time_horizon = len(rewards)
    returns = np.zeros(time_horizon, np.float32)
    advantages = np.zeros(time_horizon, np.float32)
    delta = np.zeros(time_horizon, np.float32)

    if next_is_terminal[-1]:
        returns[-1] = rewards[-1]
        delta[-1] = rewards[-1] - values[-1]
    else:
        returns[-1] = rewards[-1] + discount_factor * bootstrap_value
        delta[-1] = rewards[-1] + discount_factor * bootstrap_value - values[-1]

    advantages[-1] = delta[-1]

    for t in reversed(range(time_horizon - 1)):
        if next_is_terminal[t]:
            returns[t] = rewards[t]
            delta[t] = rewards[t] - values[t]
            advantages[t] = delta[t]
        else:
            returns[t] = rewards[t] + discount_factor * returns[t + 1]
            delta[t] = rewards[t] + discount_factor * values[t + 1] - values[t]
            advantages[t] = delta[t] + discount_factor * lam * advantages[t + 1]

    return returns, advantages


def test():
    is_terminal = np.array([False] * 10)
    #is_terminal[4] = True
    #is_terminal[9] = True
    rewards = np.ones(10)
    values = np.random.random(10)
    returns, advantages = expectations(rewards, values, is_terminal)
    advantages2 = general_advantage_estimation(rewards, values)
    advantages3 = generalized_advantage_estimation(rewards, values)
    print('LUL')


if __name__ == '__main__':
    test()

