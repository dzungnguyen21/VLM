import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    LlamaForCausalLM, LlamaTokenizer,
    get_linear_schedule_with_warmup
)
from PIL import Image
import json
import os
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Architecture ---
class LLaVAModel(nn.Module):
    def __init__(
        self,
        vision_model_name="openai/clip-vit-large-patch14",
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        projection_dim=2048,
        device_map="auto",
        load_in_8bit=False
    ):
        super().__init__()
        
        logger.info(f"Initializing LLaVA Model with Vision: {vision_model_name}, LLM: {llm_model_name}")

        # Vision encoder (CLIP)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        # Freeze vision encoder
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        
        # Get vision feature dimension
        vision_hidden_size = self.vision_tower.config.hidden_size
        
        # Projection layer to map vision features to LLM dimension
        self.mm_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Language model
        # Note: device_map="auto" and load_in_8bit might need bitsandbytes and accelerate
        kwargs = {"device_map": device_map}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            
        self.llm = LlamaForCausalLM.from_pretrained(
            llm_model_name,
            **kwargs
        )
        
        # Tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_model_name)
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Special token for image
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>") # Will be updated if added later
        
    def encode_images(self, images):
        """Encode images using CLIP vision encoder"""
        with torch.no_grad():
            image_features = self.vision_tower(images).last_hidden_state
        # Project to LLM dimension
        image_features = self.mm_projector(image_features)
        return image_features
    
    def forward(self, input_ids, attention_mask, images=None, labels=None):
        if images is not None:
            # Encode images
            image_features = self.encode_images(images)  # [B, Seq, Dim]
            
            # Find positions of <image> tokens
            batch_size = input_ids.shape[0]
            new_input_embeds = []
            new_labels = [] if labels is not None else None
            
            for i in range(batch_size):
                # Get text embeddings
                text_embeds = self.llm.model.embed_tokens(input_ids[i])
                
                # Find image token position
                # We need to make sure we are looking for the right token ID after resizing embeddings if that happened
                image_token_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
                
                if len(image_token_positions) > 0:
                    # Replace <image> token with actual image features
                    # current logic handles one image per example nicely.
                    img_pos = image_token_positions[0]
                    
                    # Concatenate: text before image + image features + text after image
                    new_embed = torch.cat([
                        text_embeds[:img_pos],
                        image_features[i],
                        text_embeds[img_pos+1:]
                    ], dim=0)
                    
                    new_input_embeds.append(new_embed)
                    
                    if labels is not None:
                        # Adjust labels for image features
                        # We mask image features in labels (set to -100)
                        new_label = torch.cat([
                            labels[i][:img_pos],
                            torch.full((image_features.shape[1],), -100, device=labels.device, dtype=labels.dtype),
                            labels[i][img_pos+1:]
                        ], dim=0)
                        new_labels.append(new_label)
                else:
                    new_input_embeds.append(text_embeds)
                    if labels is not None:
                        new_labels.append(labels[i])
            
            # Pad sequences to same length
            # Stacking variable length tensors requires padding
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            padded_embeds = []
            padded_labels = [] if labels is not None else None
            new_attention_mask = []
            
            device = input_ids.device
            
            for i in range(batch_size):
                embed = new_input_embeds[i]
                pad_len = max_len - embed.shape[0]
                
                # Pad embeddings
                padded_embed = torch.cat([
                    embed,
                    torch.zeros(pad_len, embed.shape[1], device=device, dtype=embed.dtype)
                ], dim=0)
                padded_embeds.append(padded_embed)
                
                # Pad attention mask
                # Original attention mask is for input_ids, but now we have expanded sequence
                # We assume everything valid in new_embed is attended to (1), and padding is (0)
                attn_mask = torch.cat([
                    torch.ones(embed.shape[0], device=device),
                    torch.zeros(pad_len, device=device)
                ], dim=0)
                new_attention_mask.append(attn_mask)
                
                # Pad labels
                if labels is not None:
                    label = new_labels[i]
                    padded_label = torch.cat([
                        label,
                        torch.full((pad_len,), -100, device=device, dtype=label.dtype)
                    ], dim=0)
                    padded_labels.append(padded_label)
            
            input_embeds = torch.stack(padded_embeds)
            attention_mask = torch.stack(new_attention_mask)
            labels = torch.stack(padded_labels) if labels is not None else None
            
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return outputs

# --- Dataset Class ---
class LLaVADataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, tokenizer, max_length=512):
        self.data = []
        logger.info(f"Loading data from {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add image token if not exists (This modifies tokenizer state!)
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<image>"])
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    # ...existing code...
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_path = item['image']
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.vision_processor(images=image, return_tensors="pt")['pixel_values'][0]
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            pixel_values = torch.zeros(3, 224, 224) # Dummy
        
        conversations = item['conversations']

        # Build tokens + labels per turn (robust masking)
        input_ids_list = []
        labels_list = []

        user_prefix_ids = self.tokenizer("USER:", add_special_tokens=False).input_ids
        assistant_prefix_ids = self.tokenizer("ASSISTANT:", add_special_tokens=False).input_ids

        for conv in conversations:
            if conv['from'] == 'human':
                text = f"USER: {conv['value']}\n"
                ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids_list.extend(ids)
                labels_list.extend([-100] * len(ids))
            else:
                text = f"ASSISTANT: {conv['value']}\n"
                ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids_list.extend(ids)
                # mask prefix "ASSISTANT:" only
                prefix_len = len(assistant_prefix_ids)
                labels_list.extend([-100] * prefix_len)
                labels_list.extend(ids[prefix_len:])

        # Truncate
        if len(input_ids_list) > self.max_length:
            input_ids_list = input_ids_list[:self.max_length]
            labels_list = labels_list[:self.max_length]

        # Pad
        pad_len = self.max_length - len(input_ids_list)
        if pad_len > 0:
            input_ids_list.extend([self.tokenizer.pad_token_id] * pad_len)
            labels_list.extend([-100] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        } 

def train_epoch(model, dataloader, optimizer, scheduler, epoch, grad_accum_steps, device, save_steps, output_dir):
    model.train()
    model.vision_tower.eval() # Keep vision tower frozen
    
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            labels=labels
        )
        
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # if (step + 1) % save_steps == 0:
        #     checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}_step{step+1}.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'step': step,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': total_loss / (step + 1),
        #     }, checkpoint_path)
        #     logger.info(f"Checkpoint saved to {checkpoint_path}")
            
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="llava_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Model
    # Using TinyLlama as per notebook
    model = LLaVAModel(
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_8bit=args.load_in_8bit,
        device_map="auto" if args.load_in_8bit else None
    )
    
    # Resize token embeddings for <image>
    # Note: LLaVADataset might add tokens to tokenizer, so we need to sync
    
    # Load Dataset
    dataset = LLaVADataset(
        args.train_data, 
        model.vision_processor, 
        model.tokenizer
    )
    
    if args.limit_samples:
        dataset.data = dataset.data[:args.limit_samples]
        logger.info(f"Limiting dataset to {len(dataset)} samples for testing")
    
    # Resize model embeddings to match updated tokenizer
    model.llm.resize_token_embeddings(len(model.tokenizer))
    # Update image token id in model in case it changed
    model.image_token_id = model.tokenizer.convert_tokens_to_ids("<image>")
    
    if not args.load_in_8bit:
        model = model.to(device)
    else:
        # Move other parts to device since they are not handled by device_map="auto" of LLM
        # Also cast to float16 to match 8-bit LLM expectation
        model.vision_tower = model.vision_tower.to(device).to(torch.float16)
        model.mm_projector = model.mm_projector.to(device).to(torch.float16)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if os.name != 'nt' else 0, # Windows msg
        pin_memory=True
    )
    
    # Optimizer
    trainable_params = list(model.mm_projector.parameters())
    # If we wanted to train adapter/LoRA, we'd add those parameters here
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info("Starting training...")
    best_loss = float('inf')
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            args.grad_accum_steps,
            device,
            args.save_steps,
            args.output_dir
        )
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
        # Save epoch checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved to {best_path} (loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()
