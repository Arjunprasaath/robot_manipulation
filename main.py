import numpy as np
import robosuite as rb
import matplotlib.pyplot as plt
from vlm_controller import VLMController
import os

env = rb.make(
    env_name= "PickPlaceCan",
    robots="Panda",
    use_camera_obs=True,
    camera_names="robot0_eye_in_hand",
    has_renderer=False,
    has_offscreen_renderer=True,
    camera_heights=256,
    camera_widths=256,
    control_freq=20,
    reward_shaping = True
)

# env = rb.make(
#     env_name="PickPlaceCan",
#     robots="Panda",
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     # camera_name="agentview",
# )
obs = env.reset()
# print(f"env action space: {env.action_spec}")

action_map = {
    'backward': np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Negative x (up key) FORWARD
    'forward': np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive x (down key) BACKWARD
    'left': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Negative y (left key) LEFT
    'right': np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive y (right key) RIGHT
    'up': np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),   # UP
    'down': np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),   # DOWN
    'close gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),   # CLOSE GRIPPER
    'open gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),   # OPEN GRIPPER
}

# Initialize VLM controller
model_path = os.path.join(os.path.dirname(__file__), "models/Qwen2-VL-2B")
vlm_controller = VLMController(model_path, action_map)

# Create output directory for images
os.makedirs("vlm_control_outputs", exist_ok=True)

steps = 300
done = False
previous_action = None
cumulative_reward = 0
reward = 0

print("\nStarting VLM-controlled robot manipulation...\n")

for i in range(steps):
    # Get current observation
    img = obs['robot0_eye_in_hand_image']

    # Get VLM prediction
    action, action_name = vlm_controller.predict_action_with_feedback(
        img,
        previous_action=previous_action,
        reward=reward if i > 0 else None
    )

    # Execute action
    obs, reward, done, info = env.step(action)
    cumulative_reward += reward

    print(f"Step {i}: Action={action_name}, Reward={reward:.4f}, Cumulative={cumulative_reward:.4f}")

    # Save visualization every 5 steps
    if i % 5 == 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, origin="lower")
        plt.title(f"Step {i} | Action: {action_name} | Reward: {reward:.3f}")
        plt.axis('off')
        plt.savefig(f"vlm_control_outputs/step{i:03d}.png")
        plt.close()

    previous_action = action_name

    if done:
        print(f"\nTask completed at step {i}!")
        break

print(f"\nFinal cumulative reward: {cumulative_reward:.4f}")
print(f"Images saved in vlm_control_outputs/")
env.close()
