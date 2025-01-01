# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import typer
from typer_config import use_yaml_config
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer, TrainingArguments
from gluonts.dataset.common import FileDataset
import torch.nn as nn
from chronos import ChronosPipeline, ChronosConfig
from train import (
    ChronosDataset,
    load_model,
    get_next_path,
    is_main_process,
    log_on_main,
    has_enough_observations,
    Filter,
    get_training_job_info,
    save_training_info
)

app = typer.Typer(pretty_exceptions_enable=False)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, teacher_model=None, alpha=0.75, temperature=0.1, student_hidden_size=256, teacher_hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # Get model dtype from the student model
        model_dtype = next(self.model.parameters()).dtype
        
        # Initialize projection heads with matching dtype
        self.encoder_projection = nn.Linear(
            student_hidden_size,
            teacher_hidden_size,
        ).to(device=self.model.device, dtype=model_dtype)
        
        self.decoder_projection = nn.Linear(
            student_hidden_size,
            teacher_hidden_size,
        ).to(device=self.model.device, dtype=model_dtype)
        
    def get_model_outputs(self, model, inputs, is_teacher=False):
        """Get all relevant outputs from a model"""
        if hasattr(model, 'model'):
            device = model.model.device
            model = model.model
        else:
            device = model.device
            
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        encoder_output = outputs.encoder_last_hidden_state
        decoder_output = outputs.decoder_hidden_states[-1]
        
        # Project student outputs to teacher dimension if not teacher
        if not is_teacher:
            encoder_output = self.encoder_projection(encoder_output)
            decoder_output = self.decoder_projection(decoder_output)
        
        return {
            'encoder_output': encoder_output,
            'decoder_output': decoder_output,
            'logits': outputs.logits,
            'loss': outputs.loss if not is_teacher else None
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Combined loss computation with distillation"""
        # Get teacher outputs (with no grad)
        with torch.no_grad():
            teacher_outputs = self.get_model_outputs(self.teacher_model, inputs, is_teacher=True)
        
        # Get student outputs
        student_outputs = self.get_model_outputs(model, inputs, is_teacher=False)
        
        # 1. Task-specific loss
        labels = inputs['labels']  # Shape: [batch_size, seq_len]
        logits = student_outputs['logits']
        labels = labels.contiguous().view(-1)
        logits = logits.contiguous().view(-1, logits.size(-1))
        # # Debug prints
        # print("Labels shape:", labels.shape)
        # print("Logits shape:", logits.shape)
        # print("Unique label values:", torch.unique(labels))
        # print("Label range:", labels.min().item(), "to", labels.max().item())
        # print("Vocab size:", logits.size(-1))
        
        # Compute cross entropy loss
        task_loss = self.criterion(
            logits,
            labels,
        )
        

        
        # Soft targets loss on logits
        T = self.temperature
        # Get log softmax of student outputs with temperature
        student_lsm = F.log_softmax(student_outputs['logits'] / T, dim=-1)
        # Get softmax of teacher outputs with temperature
        teacher_probs = F.softmax(teacher_outputs['logits'] / T, dim=-1)
        
        # Compute DINO loss
        distill_loss = -(teacher_probs * student_lsm).sum(dim=-1).mean()
        
        student_encoder = F.normalize(student_outputs['encoder_output'], p=2, dim=-1)
        teacher_encoder = F.normalize(teacher_outputs['encoder_output'], p=2, dim=-1)
        student_decoder = F.normalize(student_outputs['decoder_output'], p=2, dim=-1)
        teacher_decoder = F.normalize(teacher_outputs['decoder_output'], p=2, dim=-1)
        # Hidden state losses
        B = student_encoder.size(0)
        encoder_loss = F.mse_loss(
            student_encoder,
            teacher_encoder,
            reduction='sum'
        )/B
        
        decoder_loss = F.mse_loss(
            student_decoder,
            teacher_decoder,
            reduction='sum'
        )/B
        
        # Combine all distillation losses
        total_distill_loss = (
            0.6 * distill_loss +    # Soft targets
            0.2 * encoder_loss +    # Encoder states
            0.2 * decoder_loss      # Decoder states
        )
        
        # Final loss combining task and distillation
        total_loss = (
            self.alpha * task_loss + 
            (1 - self.alpha) * total_distill_loss
        )
        total_loss = total_loss
            # Log the loss components
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({
            "task_loss": task_loss.item(),
            "distill_loss/total": total_distill_loss.item(),
            "distill_loss/soft_targets": distill_loss.item(),
            "distill_loss/encoder": encoder_loss.item(),
            "distill_loss/decoder": decoder_loss.item(),
            "loss/total": total_loss.item()
        })

        
        return (total_loss, student_outputs) if return_outputs else total_loss
    def create_optimizer(self):
        """Override to include projection heads in optimization"""
        if self.optimizer is None:
            opt_model = self.model
            
            # Add projection heads to the parameters being optimized
            opt_model.projection_heads = nn.ModuleList([
                self.encoder_projection,
                self.decoder_projection
            ])
            # Get all parameters that require gradients
            optimizer_grouped_parameters = [
                {
                    "params": [p for p in opt_model.parameters() if p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                }
            ]
            
            optimizer_cls = {
                "adamw_hf": transformers.optimization.AdamW,
                "adamw_torch": torch.optim.AdamW,
                "adamw_torch_fused": torch.optim.AdamW,
                "adamw_apex_fused": torch.optim.AdamW,
                "adamw_anyprecision": torch.optim.AdamW,
            }[self.args.optim]
            
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay,
            )

        return self.optimizer
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model including projection heads"""
        # First save the main model
        super().save_model(output_dir, _internal_call)
        
        # Save projection heads
        if output_dir is None:
            output_dir = self.args.output_dir
            
        torch.save({
            'encoder_projection': self.encoder_projection.state_dict(),
            'decoder_projection': self.decoder_projection.state_dict()
        }, os.path.join(output_dir, 'projection_heads.pt'))

@app.command()
@use_yaml_config(param_name="config")
def main(
    # Teacher model parameters
    teacher_model_path: str,
    
    # Training data parameters
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    
    # Training hyperparameters
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    
    # Model parameters
    model_id: str = "google/t5-efficient-tiny",  # Student model
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    
    # Output parameters
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    
    # Tokenizer parameters
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    
    # Learning rate schedule
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    
    # Other parameters
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    
    # Generation parameters
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    
    # Distillation parameters
    alpha: float = 0.5,  # Weight between task and distillation loss
    distill_temperature: float = 0.1,  # Temperature for knowledge distillation
    student_hidden_size: int = 256,  # Hidden size of student model
    teacher_hidden_size: int = 512,  # Hidden size of teacher model
    # Random seed
    seed: Optional[int] = None,
):
    """Main function for distillation training"""
    
    if seed is None:
        seed = random.randint(0, 2**32)
    
    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)
    
    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    
    # Process other config parameters
    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    
    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
        
    output_dir = get_next_path("distill", base_dir=output_dir, file_type="")
    
    # Load teacher model
    log_on_main("Loading teacher model", logger)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    teacher = ChronosPipeline.from_pretrained(
        teacher_model_path,
        device_map="auto",
        torch_dtype=dtype
    )
    teacher = teacher.model  # Extract the inner model for training
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()  # Set to evaluation mode
    # Load student model
    log_on_main("Initializing student model", logger)
    student = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    # Convert student model to same dtype as teacher
    student = student.to(dtype=dtype)
    
    # Create Chronos config
    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,  # Added generation parameters
        top_k=top_k,
        top_p=top_p
    )
    
    # Add config to student model
    student.config.chronos_config = chronos_config.__dict__
    
    # Load datasets
    log_on_main(f"Loading datasets for training: {training_data_paths}", logger)
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]
    
    # Create training dataset
    train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    # Create distillation trainer
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=train_dataset,
        alpha=alpha,
        temperature=distill_temperature,
        student_hidden_size=student_hidden_size,
        teacher_hidden_size=teacher_hidden_size,
    )
    
    # Train
    log_on_main("Starting distillation training", logger)
    trainer.train()
    
    # Save final model and config
    if is_main_process():
        student.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final",
            training_config=raw_training_config
        )

if __name__ == "__main__":
    app()