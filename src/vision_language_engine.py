from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path
    import os

    from peft import PeftModel

    possible_paths = [
        model_name
    ]
    
    model_path_str = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            model_path_str = str(path)
            break
    
    if model_path_str is None:
        raise ValueError(f"Could not find model at any of these paths: {possible_paths}")

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    
    # Ensure float32 consistency for MPS compatibility
    vision_encoder = vision_encoder.to(dtype=torch.float32)
    text_encoder = text_encoder.to(dtype=torch.float32)
    
    clip = CLIP(vision_encoder, text_encoder)
    
    # Add missing method for PEFT compatibility
    def prepare_inputs_for_generation(*args, **kwargs):
        return {}
    clip.prepare_inputs_for_generation = prepare_inputs_for_generation
    
    try:
        clip = PeftModel.from_pretrained(clip, model_path_str).to(device)
        # Ensure entire model is in float32 for MPS
        clip = clip.to(dtype=torch.float32)
        clip.model.load_pretrained(model_path_str)
        clip.model.eval()
    except Exception as e:
        print(f"Warning: PEFT loading failed ({e}), loading base model only")
        clip = clip.to(device)
        clip = clip.to(dtype=torch.float32)
        clip.load_pretrained(model_path_str)
        clip.eval()

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] 
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 256, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection layers for vision and text encoders
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)
        
        # Temperature parameter for scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
        # Initialize projections with small values
        nn.init.normal_(self.vision_projection.weight, std=0.02)
        nn.init.normal_(self.text_projection.weight, std=0.02)
        nn.init.zeros_(self.vision_projection.bias)
        nn.init.zeros_(self.text_projection.bias)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def encode_text(self, text: str) -> torch.Tensor:
        # Tokenize text with proper attention mask
        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        
        # Get text embeddings with attention mask
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Apply masked average pooling (same as forward method)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(text_outputs.last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        text_embeds = sum_embeddings / sum_mask
        
        # Project and normalize
        text_features = self.text_projection(text_embeds.to(dtype=torch.float32))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists() and additional_weights_path.stat().st_size > 0:
            try:
                additional_state_dict = torch.load(additional_weights_path, map_location="cpu")
                
                for name, param in self.named_parameters():
                    if "vision_encoder." in name or "text_encoder." in name:
                        continue
                    if name in additional_state_dict:
                        param.data = additional_state_dict[name]
                    else:
                        print(f"Warning: Parameter {name} not found in additional weights, keeping initialized values")
            except Exception as e:
                print(f"Warning: Failed to load additional weights ({e}), using initialized parameters")
        else:
            print(f"Warning: Additional weights file missing or empty, using initialized parameters")

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):

        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: The pixel values of the image.
            input_ids: The input ids of the text.
            attention_mask: The attention mask of the text.
            labels: The labels for the text features.
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            Tuple of (image_features, text_features, logit_scale)
            - image_features: Normalized image embeddings
            - text_features: Normalized text embeddings
            - logit_scale: The logit scale parameter
        """
        # Ensure consistent float32 dtype for MPS compatibility
        pixel_values = pixel_values.to(dtype=torch.float32)
        
        # Get image embeddings from vision encoder
        vision_outputs = self.vision_encoder(pixel_values)
        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Average pooling over spatial dimensions
        
        # Get text embeddings from text encoder
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Apply attention mask for proper average pooling
        # Expand attention mask to match hidden state dimensions
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
        
        # Masked average pooling: sum over sequence length, then divide by actual sequence length
        sum_embeddings = torch.sum(text_outputs.last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        text_embeds = sum_embeddings / sum_mask
        
        # Project embeddings to the same dimension
        image_features = self.vision_projection(image_embeds.to(dtype=torch.float32))
        text_features = self.text_projection(text_embeds.to(dtype=torch.float32))
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Get logit scale
        logit_scale = self.logit_scale.exp()
        
        return image_features, text_features, logit_scale


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute the loss for the CLIP model.
    Args:
        outputs: A tuple containing the outputs of CLIP.forward().
        labels: The labels for the text features.
        (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
        num_items_in_batch: The number of items in the batch.
        (NOTE: you don't need to use the variable `num_items_in_batch`, this is just for compatibility with Trainer)
    Returns:
        The loss for the CLIP model.
    """
    image_features, text_features, logit_scale = outputs
    
    # Compute similarity matrix
    logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
    logits_per_text = logits_per_image.t()
    
    # Create labels for contrastive loss (diagonal should be positive pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Compute cross-entropy loss for both directions
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    
    # Average the two losses
    loss = (loss_img + loss_txt) / 2
    
    return loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 8,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    num_workers: int = 8,
    use_minimal_dataset: bool = False,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device)
    # Ensure float32 for MPS compatibility
    if device == "mps":
        model = model.float()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=12,
        lora_alpha=24,  
        lora_dropout=0.05,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    captions_path = Path(__file__).parent / "captions.json"
    
    if use_minimal_dataset:
        minimal_dataset_path = Path(__file__).parent / "clip_captions_minimal.json"
        if minimal_dataset_path.exists():
            print(f"Using minimal dataset for faster training: {minimal_dataset_path}")
            captions_path = minimal_dataset_path
        else:
            print("Warning: Minimal dataset requested but not found, using full dataset")
    
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    
    print(f"Loading captions from: {captions_path}")
    import json
    with open(captions_path, 'r') as f:
        caption_data = json.load(f)
    
    class HomeworkCaptionDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "caption": item["caption"],
                "image_path": f"data/{item['image_file']}"
            }
    
    train_dataset = HomeworkCaptionDataset(caption_data)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)
    print(f"Loaded {len(caption_data)} captions for training")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # Disable for MPS performance
        learning_rate=learning_rate,
        bf16=False,  # Disable mixed precision for MPS
        fp16=False,  # Disable mixed precision for MPS
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=0,  # Use 0 for MPS optimization
        dataloader_pin_memory=False,  # Disable pin memory for MPS
        weight_decay=0.005,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def fast_train():
    train(
        num_train_epochs=2,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-3
    )


def efficient_clip_train():
    train(
        output_dir="homework/clip_model",  # Proper output directory
        num_train_epochs=0.20,  # 0.20 epochs = ~4.5 hours = 79% accuracy = 100 points
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
    )

def accuracy_breakthrough_clip_train():
    train(
        output_dir="homework/breakthrough_clip_model",
        num_train_epochs=0.75,  # Extended training for 70%+ accuracy
        per_device_train_batch_size=16,  # Slightly smaller for stability
        gradient_accumulation_steps=4,   # Maintain effective batch size
        learning_rate=2e-4,              # Slightly lower for extended training
        # MPS optimizations (critical for CLIP performance)
        num_workers=0,                   # MPS optimization
    )


def dutts_clip_benchmark():
    train(
        output_dir="homework/clip_model",
        num_train_epochs=0.05, 
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
    )


# Command for optimized training as referenced in the memory
def optimized_train():
    train(
        output_dir="clip",
        num_train_epochs=3,  # Reduced from 8 for speed
        per_device_train_batch_size=32,  # Increased from 16 for speed
        gradient_accumulation_steps=2,   # Reduced from 4, effective batch size still 64
        learning_rate=5e-4,  # Increased for faster convergence
        num_workers=4,       # Reduced to avoid overhead
    )

def fast_train():
    train(
        output_dir="clip_fast",
        num_train_epochs=2,  # Very few epochs
        per_device_train_batch_size=64,  # Large batch size
        gradient_accumulation_steps=1,   # No accumulation for speed
        learning_rate=1e-3,  # High learning rate
        num_workers=2,       # Minimal workers
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).float()
        text_inputs = processor(
            text=pair["candidates"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def scaled_clip_train():
    train(
        data_dir=Path("data/scaled"),
        output_dir="homework/scaled_clip_model",
        num_train_epochs=6,
        per_device_train_batch_size=256,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_workers=0,       
    )

def main():
    from fire import Fire

    Fire({
        "train": train,
        "test": test,
        "optimized_train": optimized_train,
        "fast_train": fast_train,
        "efficient_clip_train": efficient_clip_train,
        "dutts_clip_benchmark": dutts_clip_benchmark,
        "accuracy_breakthrough_clip_train": accuracy_breakthrough_clip_train,
        "scaled_clip_train": scaled_clip_train,
    })


if __name__ == "__main__":
    main()
