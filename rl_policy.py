import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Tuple


class VLMRLPolicy(nn.Module):
    """VLM-based RL Policy for robot control with PPO."""

    def __init__(self, model_path: str, action_map: Dict[str, np.ndarray], use_lora: bool = True, device: str = None):
        super().__init__()

        # Auto-detect device: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # MPS has compatibility issues with some operations, use cautiously
                self.device = "cpu"  # Default to CPU for stability
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.action_map = action_map
        self.action_names = list(action_map.keys())
        self.num_actions = len(self.action_names)

        print(f"Loading VLM for RL training on device: {self.device}")

        # Determine dtype and device_map based on device
        if self.device == "cuda":
            dtype = torch.float16  # Use FP16 on GPU for efficiency
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = None

        # Load the base model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map
        )

        # Apply LoRA for efficient fine-tuning
        if use_lora:
            lora_config = LoraConfig(
                r=16,  # LoRA rank
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA applied. Trainable parameters: {self.model.print_trainable_parameters()}")

        self.processor = AutoProcessor.from_pretrained(model_path)

        # Add value head for PPO
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Action head for direct action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_actions)
        )

        # Move heads to device (model is already on device via device_map)
        if self.device != "cuda":  # device_map="auto" handles CUDA
            self.value_head = self.value_head.to(self.device)
            self.action_head = self.action_head.to(self.device)
        else:
            # For CUDA, move to the same device as the model
            self.value_head = self.value_head.to(self.device)
            self.action_head = self.action_head.to(self.device)

        print("VLM RL Policy initialized!")

    def prepare_inputs(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare image inputs for the model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Create prompt for vision-language understanding
        prompt = f"""Robot control task: Pick up the can and place it in a different location.
Available actions: {', '.join(self.action_names)}
Choose the best action based on the current view.
Answer:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # Fallback if text is empty
        if not text or len(text) == 0:
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs.to(self.device)

    def forward(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Returns:
            action_logits: Action logits for policy
            value: State value estimate
            hidden_state: Last hidden state for feature extraction
        """
        inputs = self.prepare_inputs(image)

        # Get model outputs
        outputs = self.model(**inputs, output_hidden_states=True)

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        # Average pool over sequence dimension
        pooled_hidden = hidden_states.mean(dim=1)  # (batch_size, hidden_size)

        # Get action logits and value
        action_logits = self.action_head(pooled_hidden)
        value = self.value_head(pooled_hidden)

        return action_logits, value, pooled_hidden

    def get_action_and_value(self, image: np.ndarray, action: int = None) -> Tuple:
        """
        Get action, log probability, entropy, and value for PPO.

        Args:
            image: Current observation
            action: If provided, compute log prob for this action

        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: State value estimate
        """
        action_logits, value, _ = self.forward(image)

        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        # Sample action if not provided
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, image: np.ndarray) -> torch.Tensor:
        """Get only the value estimate for an observation."""
        _, value, _ = self.forward(image)
        return value.squeeze(-1)

    def select_action(self, image: np.ndarray, deterministic: bool = False) -> Tuple[int, str]:
        """
        Select an action based on the current observation.

        Args:
            image: Current observation
            deterministic: If True, select argmax action

        Returns:
            action_idx: Index of selected action
            action_name: Name of selected action
        """
        with torch.no_grad():
            action_logits, _, _ = self.forward(image)

            if deterministic:
                action_idx = torch.argmax(action_logits, dim=-1).item()
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()

        action_name = self.action_names[action_idx]
        return action_idx, action_name

    def save(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'action_head_state_dict': self.action_head.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.action_head.load_state_dict(checkpoint['action_head_state_dict'])
        print(f"Model loaded from {path}")
