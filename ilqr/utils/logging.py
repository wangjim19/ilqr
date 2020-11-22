import numpy as np

def verbose_iteration_callback(iteration_count, xs, us, J_opt, accepted, converged, rollout_history = None):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
    if rollout_history is not None:
        if "observations" in rollout_history:
            rollout_history["observations"] = np.concatenate((rollout_history["observations"], xs[:-1]))
            rollout_history["actions"] = np.concatenate((rollout_history["actions"], us))
            rollout_history["next_observations"] = np.concatenate((rollout_history["next_observations"], xs[1:]))
        else:
            rollout_history["observations"] = xs[:-1]
            rollout_history["actions"] = us
            rollout_history["next_observations"] = xs[1:]

def cost_only_callback(iteration_count, xs, us, J_opt, accepted, converged, rollout_history = None):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)
    if rollout_history is not None:
        if "observations" in rollout_history:
            rollout_history["observations"] = np.concatenate((rollout_history["observations"], xs[:-1]))
            rollout_history["actions"] = np.concatenate((rollout_history["actions"], us))
            rollout_history["next_observations"] = np.concatenate((rollout_history["next_observations"], xs[1:]))
        else:
            rollout_history["observations"] = xs[:-1]
            rollout_history["actions"] = us
            rollout_history["next_observations"] = xs[1:]
