# test.py
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.camera_only.camera_only_agent import CameraOnlyAgent

if __name__ == "__main__":
    try:
        trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
        agent = CameraOnlyAgent(lr=1e-4, trajectory_sampling=trajectory_sampling)
        print("Successfully instantiated the agent!")
    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"Other error: {e}")