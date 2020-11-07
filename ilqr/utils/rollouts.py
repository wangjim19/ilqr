import numpy as np

def monitored_rollout(dynamics, initial_state, controls):
	"""
		rollout given
			dynamics: MujocoDynamics
			initial state: np.ndarray
			control sequence: list[np.ndarray]
		returns
			trajectory: list[np.ndarray]
			video_frames: list[np.ndarray]
	"""
	dynamics.set_state(initial_state)
	
	trajectory = [initial_state.copy()]
	video_frames = [dynamics.render()]

	for i, control in enumerate(controls):
	    next_state = dynamics.step(control)
	    img = dynamics.render()

	    trajectory.append(next_state)
	    video_frames.append(img)

	return trajectory, video_frames