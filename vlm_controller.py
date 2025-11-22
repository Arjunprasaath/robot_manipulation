import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

class VLMController:
    def __init__(self, model_path, action_map):
        """
        Initialize the VLM controller.

        Args:
            model_path: Path to the Qwen2-VL model
            action_map: Dictionary mapping action names to numpy arrays
        """
        print("Loading VLM model...")
        # Use CPU for stable inference (MPS has compatibility issues)
        self.device = "cpu"
        print(f"Using device: {self.device}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.action_map = action_map
        self.action_names = list(action_map.keys())

        print("VLM model loaded successfully!")

    def predict_action(self, image, context=""):
        """
        Predict the next robot action based on the current image.

        Args:
            image: numpy array of shape (H, W, 3) representing the camera image
            context: optional context string to guide the prediction

        Returns:
            action: numpy array representing the robot action
            action_name: string name of the predicted action
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Create the prompt
        action_options = ", ".join(self.action_names)
        prompt = f"""Robot control task: Pick up the can in the environment.
Available actions: {action_options}
Details on action:
    forward: move the robot gripper forward
    backward: move the robot gripper backward
    left: move the robot gripper to the left
    right:  move the robot gripper to the right
    up: move the robot gripper up (z axis)
    down:  move the robot gripper down (z axis but down)
    close gripper: close the gripper a little bit
    open gripper: open the gripper a little bit
Based on the image, choose exactly ONE action, each action does incremental change so until you pick up the can keep taking actions.
Answer:"""

        # Prepare messages in the format expected by Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process the input using qwen-vl-utils
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # If text is empty, manually construct it
        if not text or len(text) == 0:
            # Fallback: construct text manually with vision tokens
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = inputs.to(self.device)

        # Generate prediction
        with torch.no_grad():
            try:
                # Use greedy decoding with limited tokens
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Limit to prevent long generations
                    do_sample=False,  # Greedy decoding
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            except Exception as e:
                print(f"Generation error: {e}")
                # Fallback to default action
                return self.action_map["forward"], "forward"

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse the output to get action
        action_name = self._parse_action(output_text)
        action = self.action_map.get(action_name, self.action_map["forward"])

        print(f"VLM output: {output_text.strip()}")
        print(f"Predicted action: {action_name}")

        return action, action_name

    def _parse_action(self, output_text):
        """
        Parse the VLM output to extract the action name.
        """
        output_lower = output_text.lower().strip()

        # Try exact match first
        for action_name in self.action_names:
            if action_name.lower() == output_lower:
                return action_name

        # Try partial match
        for action_name in self.action_names:
            if action_name.lower() in output_lower:
                return action_name

        # Default to forward if no match
        print(f"Warning: Could not parse action from '{output_text}', defaulting to 'forward'")
        return "forward"

    def predict_action_with_feedback(self, image, previous_action=None, reward=None):
        """
        Predict action with feedback from previous step.

        Args:
            image: Current camera image
            previous_action: Name of the previous action taken
            reward: Reward received from the environment

        Returns:
            action: numpy array representing the robot action
            action_name: string name of the predicted action
        """
        context = ""
        if previous_action and reward is not None:
            context = f"\nPrevious action: {previous_action}\nReward: {reward:.3f}"

        return self.predict_action(image, context)
