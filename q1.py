import numpy as np
import pandas as pd

def to_log(p):
    return float(np.log10(p)) * -1

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = []
    path = []

    V.append([])

    for idx, y in enumerate(states):
        V[0].append(to_log(start_p[y]) + to_log(emit_p[y][obs[0]]))
        path.append([idx])

    for t in range(1, len(obs)):
        V.append([])
        newpath = []

        for idx, y in enumerate(states):
            probs = [V[t-1][idx0] + to_log(trans_p[y0][y]) + to_log(emit_p[y][obs[t]]) for idx0, y0 in enumerate(states)]
            best_prob, best_state = float(np.min(probs)), np.argmin(probs)
            V[t].append(best_prob)
            newpath.append(path[best_state] + [idx])

        path = newpath

    final_probs = [V[-1][idx] for idx in range(len(states))]
    best_prob, best_state = float(np.min(final_probs)), np.argmin(final_probs)

    best_path = [states[idx] for idx in path[best_state]]

    log_probabilities_matrix = pd.DataFrame(V)
    log_probabilities_matrix.rename_axis("Time", axis=1, inplace=True)

    for idx, state in enumerate(states):
        log_probabilities_matrix.rename(columns={idx: f"Log Probability {state}"}, inplace=True)

    log_probabilities_matrix["Char"] = obs
    log_probabilities_matrix["State"] = best_path

    return (best_prob, best_path, log_probabilities_matrix)

states = ['L', 'R']
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
print("Log Probabilities Matrix: \n", log_probabilities_matrix)
print("Log Probability: ", prob)
print("Best Decoding Path: ", path)