import os
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from dataset import get_cc3m_dataloader
from model import CoLLMStage1
from loss import collm_loss
from slerp import slerp
from text_synthesis import get_modification_texts

def get_nearest_neighbors(h_prime_detached):
    sim_matrix = h_prime_detached @ h_prime_detached.T
    sim_matrix.fill_diagonal_(-float("inf"))
    nn_indices = sim_matrix.argmax(dim=1)
    return h_prime_detached[nn_indices], nn_indices

def main():
    parser = argparse.ArgumentParser()
    # Note: Use shell expansion syntax for WebDataset
    parser.add_argument("--data_pattern", type=str, required=True, 
                        help="e.g. /home/rahul/shyam/aditya/training/cc3m_downloaded_80k_224/{00000..00011}.tar")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--llm_dim", type=int, default=1024) # Change to 4096 if switching to SFR
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    dataloader = get_cc3m_dataloader(args.data_pattern, batch_size=args.batch_size)

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")
    
    model = CoLLMStage1(
        tokenizer=tokenizer, 
        llm_model_name=args.llm_model, 
        llm_dim=args.llm_dim
    ).to(device)
    model.train()

    # 3. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    print("Starting training...")
    # Since WebDataset streams, we just loop over it
    for step, batch in enumerate(dataloader):
        aug_images = batch["aug_images"].to(device)
        target_images = batch["target_images"].to(device)
        captions = batch["captions"]
        
        # 1. Target Branch (DETACHED)
        with torch.no_grad():
            z = model.encode_image_features(target_images)

        # 2. Augmented Branch (WITH GRADIENTS for CLIP)
        h_prime = model.encode_image_features(aug_images)

        # 3. Nearest Neighbor (DETACHED)
        with torch.no_grad():
            nn_embeds, nn_indices = get_nearest_neighbors(h_prime.detach())
        captions_j = [captions[i] for i in nn_indices]

        # 4. Slerp & Synthesis
        h_star = slerp(h_prime, nn_embeds, t=0.5)
        mod_texts = get_modification_texts(captions, captions_j, synthesis_ratio=0.75)

        # 5. Get Queries
        c_v, c_w, c = model.forward_queries(h_star, captions, mod_texts, device)

        # 6. Loss & Backprop
        loss, loss_dict = collm_loss(c_v, c_w, c, z, model.logit_scale)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | Temperature {torch.exp(model.logit_scale).item():.2f} | L_v {loss_dict['loss_image_only']:.4f}")

if __name__ == "__main__":
    main()