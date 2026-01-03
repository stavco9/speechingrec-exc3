import numpy as np
import pandas as pd

def set_log_probability(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = set_log_probability(v)
        elif isinstance(v, float):
            v = float(np.log10(v)) * -1
        d[k] = v
    return d

def to_log(p):
    return float(np.log10(p))

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for y in states:
        V[0][y] = to_log(start_p[y]) + to_log(emit_p[y][obs[0]])
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                [(V[t-1][y0] + to_log(trans_p[y0][y]) + to_log(emit_p[y][obs[t]]), y0) for y0 in states]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath

    (prob, state) = max([(V[-1][y], y) for y in states])

    log_probabilities_matrix = pd.DataFrame(V)
    log_probabilities_matrix["Char"] = obs

    return (prob, path[state], log_probabilities_matrix)

states = ('L', 'R')
observations = "c c b a b d c a a c d b b".split()
start_probability = {'L': 0.5, 'R': 0.5}
transition_probability = {
   'L' : {'L': 0.55, 'R': 0.45},
   'R' : {'L': 0.4, 'R': 0.6},
   }
emission_probability = {
   'L' : {'a': 0.15, 'b': 0.3, 'c': 0.35, 'd': 0.2},
   'R' : {'a': 0.35, 'b': 0.2, 'c': 0.15, 'd': 0.3},
   }

prob, path, log_probabilities_matrix = viterbi(observations, states, start_probability, transition_probability, emission_probability)
print("Probability: ", prob)
print("Path: ", path)
print("Log Probabilities Matrix: \n", log_probabilities_matrix)
