import numpy as np

alg_colors = {
    "unconstrained": "green",
    "LR": "#AA5D1F",
    "RSPO": "#BA2DC1",
    "SQRL": "#D43827",
    "RP": "#4899C5",
    "RCPO": "#34539C",
    "LBAC": "#60CC38",
}

alg_names = {
    "unconstrained": "Unconstrained",
    "LR": "LR",
    "RSPO": "RSPO",
    "SQRL": "SQRL",
    "RP": "RP",
    "RCPO": "RCPO",
    "LBAC": "Ours: LBAC",
}


def get_color(algname, alt_color_map={}):
    if algname in alg_colors:
        return alg_colors[algname]
    elif algname in alt_color_map:
        return alt_color_map[algname]
    else:
        return np.random.rand(3, )


def get_legend_name(algname, alt_name_map={}):
    if algname in alg_names:
        return alg_names[algname]
    elif algname in alt_name_map:
        return alt_name_map[algname]
    else:
        return algname


def get_stats(data):
    minlen = min([len(d) for d in data])
    data = [d[:minlen] for d in data]
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(len(data))
    ub = mu + np.std(data, axis=0) / np.sqrt(len(data))
    return mu, lb, ub


def moving_average(x, N):
    window_means = []
    for i in range(len(x) - N + 1):
        window = x[i : i + N]
        num_nans = np.count_nonzero(np.isnan(window))
        window_sum = np.nansum(window)
        if num_nans < N:
            window_mean = window_sum / (N - num_nans)
        else:
            window_mean = np.nan
        window_means.append(window_mean)
    return window_means