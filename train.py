import os
import argparse
import json
import torch
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

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
    # 1. Path directly matched to your directory structure
    parser.add_argument("--data_pattern", type=str, 
                        default="/home/rahul/shyam/aditya/training/cc3m_downloaded_80k_224/{00000..00011}.tar")
    #Changed for SFR-Embedding-2R
    parser.add_argument("--llm_model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Directories and TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    step_logs = []

    # 3. Dataset
    dataloader = get_cc3m_dataloader(args.data_pattern, batch_size=args.batch_size)

    # 4. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")
    
    model = CoLLMStage1(
        tokenizer=tokenizer, 
        llm_model_name=args.llm_model, 
        llm_dim=args.llm_dim
    ).to(device)
    model.train()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")

    # 5. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("Starting training...")
    
    # 6. Training Loop (WebDataset loops until tar files are exhausted)
    for step, batch in enumerate(dataloader):
        aug_images = batch["aug_images"].to(device)
        target_images = batch["target_images"].to(device)
        captions = batch["captions"]
        
        # 1. Target Branch (DETACHED)
        # with torch.no_grad():
        #Changed as new encodings
        z = model.encode_target(target_images)

        h_prime = model.encode_reference(aug_images)

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

        # 7. Logging to TensorBoard and JSON
        temperature_val = torch.exp(model.logit_scale).item()

        writer.add_scalar("Loss/Total", loss_dict["loss"], step)
        writer.add_scalar("Loss/Image_Only_cv", loss_dict["img_only"], step)
        writer.add_scalar("Loss/Text_Only_cw", loss_dict["txt_only"], step)
        writer.add_scalar("Loss/Composed_c", loss_dict["comp"], step)
        writer.add_scalar("Metrics/Temperature", temperature_val, step)
        
        step_logs.append({
            "step": step,
            "loss": loss_dict["loss"],
            "loss_cv": loss_dict["img_only"],
            "loss_cw": loss_dict["txt_only"],
            "loss_c": loss_dict["comp"],
            "temperature": temperature_val
        })

        if step % 10 == 0:
            print(
                f"Step {step:4d} | "
                f"Loss {loss_dict['loss']:.4f} | "
                f"L_v {loss_dict['img_only']:.4f} | "
                f"L_w {loss_dict['txt_only']:.4f} | "
                f"L_c {loss_dict['comp']:.4f} | "
                f"Temp {temperature_val:.2f}",
                flush=True
            )

        # Save checkpoint periodically (every 500 steps ~ 32,000 images)
        if step > 0 and step % 500 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")
            # torch.save(
            #     {
            #         "step": step,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #     },
            #     ckpt_path,
            # )
            #Change:
            torch.save(
                {
                    "step": step,
                    "clip_lora": model.clip.state_dict(),        # only LoRA layers inside
                    "llm_lora": model.llm.state_dict(),          # only LoRA layers inside
                    "image_adapter": model.image_adapter.state_dict(),
                    "projection_head": model.projection_head.state_dict(),
                    "logit_scale": model.logit_scale.item(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved Checkpoint -> {ckpt_path}")

    # 8. End of Training Cleanup
    final_ckpt = os.path.join(args.output_dir, "checkpoint_final.pt")
    # torch.save(model.state_dict(), final_ckpt)
    #Change:
    torch.save(
        {
            "clip_lora": model.clip.state_dict(),
            "llm_lora": model.llm.state_dict(),
            "image_adapter": model.image_adapter.state_dict(),
            "projection_head": model.projection_head.state_dict(),
            "logit_scale": model.logit_scale.item(),
        },
        final_ckpt,
    )
    print(f"\nFinal Checkpoint Saved -> {final_ckpt}")

    with open(os.path.join(args.log_dir, "training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=4)
    print(f"Logs saved -> {args.log_dir}/training_logs.json")

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()