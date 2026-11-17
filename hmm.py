import numpy as np

states = ["байгаа", "байхгүй"]
start = np.array([0.5, 0.5])
trans = np.array([
    [0.9, 0.1],
    [0.15, 0.85],
])

motion_names = ["хөдөлгөөнгүй", "хөдөлгөөнтэй"]
motion_prob = np.array([
    [0.35, 0.65],
    [0.95, 0.05],
])

temp_names = ["буурсан", "тогтвортой", "өссөн"]
temp_prob = np.array([
    [0.2, 0.5, 0.3],
    [0.45, 0.45, 0.1],
])

time_names = ["шөнө", "өглөө", "өдөр", "орой"]
time_prob = np.array([
    [0.35, 0.2, 0.1, 0.35],
    [0.1, 0.2, 0.55, 0.15],
])


def emission(obs):
    m, t, h = obs
    return motion_prob[:, m] * temp_prob[:, t] * time_prob[:, h]


def forward(observations):
    alpha = start * emission(observations[0])
    alpha = alpha / alpha.sum()
    result = [alpha]
    for obs in observations[1:]:
        alpha = (alpha @ trans) * emission(obs)
        alpha = alpha / alpha.sum()
        result.append(alpha)
    return np.array(result)


def viterbi(observations):
    n = len(observations)
    delta = np.log(start) + np.log(emission(observations[0]))
    back = np.zeros((n, 2), dtype=int)
    for i in range(1, n):
        scores = delta[:, None] + np.log(trans)
        back[i] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + np.log(emission(observations[i]))
    path = [int(delta.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def sample(n, rng):
    hidden = []
    observations = []
    s = rng.choice(2, p=start)
    for _ in range(n):
        hidden.append(s)
        m = rng.choice(2, p=motion_prob[s])
        t = rng.choice(3, p=temp_prob[s])
        h = rng.choice(4, p=time_prob[s])
        observations.append((m, t, h))
        s = rng.choice(2, p=trans[s])
    return hidden, observations


if __name__ == "__main__":
    example = [(1, 2, 0), (0, 1, 0), (0, 1, 1), (0, 0, 2), (0, 0, 2), (1, 1, 3), (1, 2, 3), (0, 1, 0)]
    probs = forward(example)
    path = viterbi(example)
    print(f"{'алхам':>5} | {'хөдөлгөөн':<13} {'температур':<11} {'цаг':<6} | {'P(байгаа)':>9} | Viterbi")
    for i, (m, t, h) in enumerate(example):
        print(f"{i + 1:>5} | {motion_names[m]:<13} {temp_names[t]:<11} {time_names[h]:<6} | {probs[i][0]:>9.3f} | {states[path[i]]}")

    rng = np.random.default_rng(0)
    hidden, observations = sample(1000, rng)
    hidden = np.array(hidden)
    filtered = np.where(forward(observations)[:, 0] > 0.5, 0, 1)
    decoded = np.array(viterbi(observations))
    motion_only = np.array([0 if m == 1 else 1 for m, _, _ in observations])
    print()
    print(f"Зөвхөн хөдөлгөөн мэдрэгч: {np.mean(motion_only == hidden):.3f}")
    print(f"HMM forward:              {np.mean(filtered == hidden):.3f}")
    print(f"HMM Viterbi:              {np.mean(decoded == hidden):.3f}")
