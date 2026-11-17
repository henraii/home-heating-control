import numpy as np

temps = list(range(14, 27))
presence = [0, 1]
actions = ["бууруулах", "хэвийн", "нэмэгдүүлэх"]
states = [(t, p) for t in temps for p in presence]

effect = {
    0: {-1: 0.7, 0: 0.3},
    1: {-1: 0.2, 0: 0.8},
    2: {1: 0.4, 2: 0.6},
}
energy = [0, 1, 5]
stay = 0.95
comfort_low = 20
comfort_high = 23
safe_low = 16
gamma = 0.8


def clamp(t):
    return min(max(t, temps[0]), temps[-1])


def reward(state, action):
    t, p = state
    if p == 1:
        if comfort_low <= t <= comfort_high:
            comfort = 10
        else:
            comfort = -2 * min(abs(t - comfort_low), abs(t - comfort_high))
    else:
        comfort = -5 if t < safe_low else 0
    return comfort - energy[action]


def transitions(state, action):
    t, p = state
    result = []
    for dt, pt in effect[action].items():
        for p2 in presence:
            pp = stay if p2 == p else 1 - stay
            result.append((pt * pp, (clamp(t + dt), p2)))
    return result


def step(state, action, rng):
    outcomes = transitions(state, action)
    probs = [o[0] for o in outcomes]
    i = rng.choice(len(outcomes), p=probs)
    return outcomes[i][1], reward(state, action)


def q_value(V, state, action):
    return reward(state, action) + gamma * sum(pr * V[s2] for pr, s2 in transitions(state, action))


def value_iteration(theta=1e-4):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            best = max(q_value(V, s, a) for a in range(len(actions)))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    policy = {s: int(np.argmax([q_value(V, s, a) for a in range(len(actions))])) for s in states}
    return V, policy


if __name__ == "__main__":
    V, policy = value_iteration()
    print(f"{'T':>4} | {'хүн байхгүй':<14} {'V':>7} | {'хүн байгаа':<14} {'V':>7}")
    for t in temps:
        a0, a1 = policy[(t, 0)], policy[(t, 1)]
        print(f"{t:>4} | {actions[a0]:<14} {V[(t, 0)]:>7.2f} | {actions[a1]:<14} {V[(t, 1)]:>7.2f}")
