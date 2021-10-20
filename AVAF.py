# -*- coding: utf-8 _*_
# @Time : 17/10/2021 4:15 pm
# @Author: ZHA Mengyue
# @FileName: AVAF.py
# @Software: Blackjack
# @Blog: https://github.com/Dolores2333

import numpy as np


# areturn the Q value = np.dot(feature, w)
def Q_function(key, w):
    player, dealer, action = key
    feature = np.array([player, dealer, action])
    return np.dot(feature, w)


# Calculate the whole Q_sa table by Q value = np.dot(feature, w)
def calculate_Q_sa(Q_sa, w):
    # Update the whole Q table
    for key in Q_sa:
        player, dealer, action = key
        feature = np.array([player, dealer, action])
        Q_sa[key] = np.dot(feature, w)
    return Q_sa


# Update the w according to MC linear AVAF
def MC_linear(reward_list, keys_by_players, N_s, N_sa, alpha, w):
    for i in range(len(reward_list)):
        if reward_list[i] is not None:
            for key in keys_by_players[i]:
                N_s[key[:-1]] += 1
                N_sa[key] += 1
                player, dealer, action = key
                feature = np.array([player, dealer, action])
                Q_hat = np.dot(feature, w)
                Q_true = reward_list[i]
                w += alpha * (Q_true-Q_hat) * feature

    return w


# Update the w according to QL linear AVAF
def QL_linear(reward_list, keys_by_players, N_s, N_sa, alpha, w):
    for i in range(len(reward_list)):
        if reward_list[i] is not None:
            for j, key in enumerate(keys_by_players[i]):
                N_s[key[:-1]] += 1
                N_sa[key] += 1

                Q_hat = Q_function(key, w)
                player, dealer, action = key
                feature = np.array([player, dealer, action])

                # Find the Q(s', a') here 0.8 is the discount factor
                if j < len(keys_by_players[i]) - 1:
                    """
                    hit = keys_by_players[i, j+1][:2] + (1, )
                    stick = keys_by_players[i, j+1][:2] + (0, )
                    """
                    keys_by_current_player = keys_by_players[i]
                    # print(keys_by_current_player)
                    next_state = keys_by_current_player[j+1][:-1]
                    hit = (next_state[0], next_state[1], 0)
                    stick = (next_state[0], next_state[1], 1)
                    """
                    print(f'This is hit in QL:{hit}')
                    print(f'This is stick in QL:{stick}')
                    """
                    Q_hit = Q_function(hit, w)
                    Q_stick = Q_function(stick, w)

                    max_value = max(Q_hit, Q_stick)
                    new = 0.8 * max_value
                else:
                    new = 0

                Q_true = reward_list[i] + new
                w += alpha * (Q_true-Q_hat) * feature
    return w


# Update the w according to TD linear AVAF
def TD_linear(reward_list, keys_by_players, N_s, N_sa, alpha, w):
    for i in range(len(reward_list)):
        if reward_list[i] is not None:
            for j, key in enumerate(keys_by_players[i]):
                N_s[key[:-1]] += 1
                N_sa[key] += 1

                # Update Q-value function
                Q_hat = Q_function(key, w)
                player, dealer, action = key
                feature = np.array([player, dealer, action])

                if j < len(keys_by_players[i]) - 1:
                    keys_by_current_player = keys_by_players[i]
                    next_key = keys_by_current_player[j+1]
                    new = 0.8 * Q_function(next_key, w)
                else:
                    new = 0
                Q_true = reward_list[i] + new
                w += alpha * (Q_true-Q_hat) * feature
    return w
