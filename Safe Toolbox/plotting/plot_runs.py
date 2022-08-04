import os
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from plotting_utils import *

# ** Set plot type here **
PLOT_TYPE = "violation"
assert PLOT_TYPE in ["ratio", "success", "violation", "PR", "reward"]


eps = {
    "drone_hover": 300,
    "drone_xz": 2000,
    "kine_car": 3000,
}

envname = {
    "drone_hover": 'DroneHover-v0',
    "drone_xz": 'DroneXZ',
    "kine_car": 'KineCar',
}

if PLOT_TYPE == "ratio":
    yscaling = {
        "drone_hover": 0.3,
        "drone_xz": 0.3,
        "kine_car": 0.1,
    }
elif PLOT_TYPE == "success":
    yscaling = {
        "drone_hover": 1,
        "drone_xz": 1,
        "kine_car": 0.5,
    }
elif PLOT_TYPE == "violation":
    yscaling = {
        "drone_hover": 0.3,
        "drone_xz": 0.3,
        "kine_car": 0.3,
    }
elif PLOT_TYPE == "reward":
    yscaling = {
        "drone_hover": 1,
        "drone_xz": 1,
        "kine_car": 0.8,
    }


def plot_experiment(experiment, logdir):
    """
        Construct experiment map for this experiment
    """
    experiment_map = {}
    experiment_map["algs"] = {}
    for fname in os.listdir(logdir):
        alg_name = fname.split("Gaussian_")[-1]
        if alg_name not in experiment_map["algs"]:
            experiment_map["algs"][alg_name] = [fname]
        else:
            experiment_map["algs"][alg_name].append(fname)

    experiment_map["outfile"] = osp.join("plotting", experiment + ".png")
    """
        Save plot for experiment
    """
    print("EXP NAME: ", experiment)
    max_eps = eps[experiment]
    fig, axs = plt.subplots(1, figsize=(16, 8))

    axs.set_title(envname[experiment], fontsize=48)
    axs.set_ylim(-0.1, int(yscaling[experiment] * max_eps) + 1)
    axs.set_xlabel("Episode", fontsize=42)
    if PLOT_TYPE == "ratio":
        axs.set_ylabel("Ratio of Successes/Violations", fontsize=42)
    elif PLOT_TYPE == "success":
        axs.set_ylabel("Cumulative Task Successes", fontsize=42)
    elif PLOT_TYPE == "violation":
        axs.set_ylabel("Cumulative Constraint Violations", fontsize=42)
        axs.set_ylim(-0.1, 2000)
    elif PLOT_TYPE == "PR":
        axs.set_ylabel("Task Successes", fontsize=42)
        axs.set_xlabel("Constraint Violations", fontsize=42)
    elif PLOT_TYPE == "reward":
        axs.set_ylabel("Reward", fontsize=42)
    else:
        raise NotImplementedError("Unsupported Plot Type")

    axs.tick_params(axis="both", which="major", labelsize=36)
    plt.subplots_adjust(hspace=0.3)
    final_ratios_dict = {}

    for alg in experiment_map["algs"]:
        print(alg)
        exp_dirs = experiment_map["algs"][alg]
        fnames = [osp.join(exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]
        print(fnames)
        task_successes_list = []
        train_rewards_safe_list = []
        train_violations_list = []

        for fname in fnames:
            with open(osp.join(logdir, fname), "rb") as f:
                data = pickle.load(f)
            train_stats = data["train_stats"]

            train_violations = []
            train_rewards = []
            task_successes = []
            for traj_stats in train_stats:
                train_violations.append([])
                train_rewards.append(0)
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats["constraint"])
                    train_rewards[-1] += -step_stats["reward"]
                    if step_stats["constraint"] > 0:
                        train_rewards[-1] += 2000
                task_successes.append(int(step_stats["success"]))

            train_violations = np.array([np.sum(t) > 0 for t in train_violations])[:max_eps]
            train_violations = np.cumsum(train_violations)

            train_rewards = np.array(train_rewards)[:max_eps]
            task_successes = np.array(task_successes)[:max_eps]
            task_successes = np.cumsum(task_successes)
            
            task_successes_list.append(task_successes)
            train_rewards_safe_list.append(train_rewards)
            train_violations_list.append(train_violations)


        task_successes_list = np.array(task_successes_list)
        train_violations_list = np.array(train_violations_list)

        # Smooth out train rewards
        for i in range(len(train_rewards_safe_list)):
            train_rewards_safe_list[i] = moving_average(train_rewards_safe_list[i], 100)

        train_rewards_safe_list = np.array(train_rewards_safe_list)

        safe_ratios = (task_successes_list + 1) / (train_violations_list + 1)
        final_ratio = safe_ratios.mean(axis=0)[-1]
        final_successes = task_successes_list[:, -1]
        final_violations = train_violations_list[:, -1]

        final_success_mean = np.mean(final_successes)
        final_success_err = np.std(final_successes) / np.sqrt(len(final_successes))
        final_violation_mean = np.mean(final_violations)
        final_violation_err = np.std(final_violations) / np.sqrt(len(final_violations))

        final_ratios_dict[alg] = final_ratio
        safe_ratios_mean, safe_ratios_lb, safe_ratios_ub = get_stats(safe_ratios)
        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trew_mean, trew_lb, trew_ub = get_stats(train_rewards_safe_list)

        if PLOT_TYPE == "ratio":
            axs.fill_between(
                range(safe_ratios_mean.shape[0]),
                safe_ratios_ub,
                safe_ratios_lb,
                color=get_color(alg),
                alpha=0.25,
                label=get_legend_name(alg),
            )
            axs.plot(safe_ratios_mean, color=get_color(alg))
        elif PLOT_TYPE == "success":
            axs.fill_between(
                range(ts_mean.shape[0]),
                ts_ub,
                ts_lb,
                color=get_color(alg),
                alpha=0.25,
                label=get_legend_name(alg),
            )
            axs.plot(ts_mean, color=get_color(alg))
        elif PLOT_TYPE == "violation":
            axs.fill_between(
                range(tv_mean.shape[0]),
                tv_ub,
                tv_lb,
                color=get_color(alg),
                alpha=0.25,
                label=get_legend_name(alg),
            )
            axs.plot(tv_mean, color=get_color(alg))
        elif PLOT_TYPE == "PR":
            axs.errorbar(
                [final_violation_mean],
                [final_success_mean],
                xerr=[final_violation_err],
                yerr=[final_success_err],
                fmt="-o",
                markersize=20,
                linewidth=5,
                color=get_color(alg),
                label=get_legend_name(alg),
            )
        elif PLOT_TYPE == "reward":
            axs.fill_between(
                range(trew_mean.shape[0]),
                trew_ub,
                trew_lb,
                color=get_color(alg),
                alpha=0.25,
                label=get_legend_name(alg),
            )
            axs.plot(trew_mean, color=get_color(alg))
        else:
            raise NotImplementedError("Unsupported Plot Type")

    print("FINAL RATIOS: ", final_ratios_dict)
    axs.legend(loc="upper left", fontsize=30, frameon=False)
    # plt.savefig(experiment_map["outfile"], bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    experiment = "drone_xz"  # ** insert experiment name here **
    logdir = "/home/desong/Shaohang/Actor_critic_with_safety/plotting/plot_training_logs" # ** insert logdir here **
    plot_experiment(experiment, logdir)
