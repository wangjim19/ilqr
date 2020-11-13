def verbose_iteration_callback(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)

def cost_only_callback(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)
