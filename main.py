import numpy as np
import robosuite as rb

# env = rb.make(
#     env_name= "PickPlaceCan",
#     robots="Panda",
#     use_camera_obs=True,
#     camera_name="agentview",
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     camera_height=84,
#     camera_width=84
# )

env = rb.make(
    env_name="PickPlaceCan",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    # camera_name="agentview",
)
env.reset()
# print(f"env action space: {env.action_spec}")

action_map = {
    'forward': np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Negative x (up key) FORWARD
    'backward': np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive x (down key) BACKWARD
    'left': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Negative y (left key) LEFT
    'right': np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive y (right key) RIGHT
    'up': np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),   # UP
    'down': np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),   # DOWN
    'close gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),   # OPEN GRIPPER
    'open gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),   # OPEN GRIPPER
}

steps = 50
done = False
for i in range(steps):
    # action = np.random.randn(*env.action_spec[0].shape)
    # random_choice = np.random.randint(0,1)
    # choice = "open gripper" if random_choice == 0 else "close gripper"
    action = action_map["forward"]
    obs, reward, done, info = env.step(action)
    env.render()
    # if done:
    #     break
