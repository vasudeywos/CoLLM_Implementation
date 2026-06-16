import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

#Changed for SFR-Embedding-2R
class ImageAdapter(nn.Module):
    def __init__(self, clip_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    def forward(self, x):
        return self.proj(x).unsqueeze(1)  # [B, 1, 1024]

#Changed for SFR-Embedding-2R
class ProjectionHead(nn.Module):
    def __init__(self, llm_dim: int = 4096, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(llm_dim, embed_dim)
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

#Changed for SFR-Embedding-2R
class CoLLMStage1(nn.Module):
    def __init__(
        self,
        tokenizer,
        clip_model_name="openai/clip-vit-large-patch14",
        llm_model_name="Salesforce/SFR-Embedding-2_R",
        lora_rank=64,
        clip_dim=1024,
        llm_dim=4096,
        embed_dim=768
    ):
        super().__init__()
        self.tokenizer = tokenizer
        
        # 1. Vision Encoder
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        clip_lora = LoraConfig(r=lora_rank, 
                               lora_alpha=lora_rank, 
                               target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                               lora_dropout=0.1,#Added 
                               bias="none")
        self.clip.vision_model = get_peft_model(self.clip.vision_model, clip_lora)
        for p in self.clip.text_model.parameters(): p.requires_grad = False
        for p in self.clip.text_projection.parameters(): p.requires_grad = False

        # 2. LLM
        # Use bfloat16 to save memory and match Qwen/SFR natives
        self.llm = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16)

        #Added:
        self.llm.gradient_checkpointing_enable()
        
        # Inject <image> token safely
        self.image_token = "<image>"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        llm_lora = LoraConfig(r=lora_rank, 
                              lora_alpha=lora_rank, 
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
                              lora_dropout=0.1,#Added
                              bias="none", 
                              task_type=TaskType.FEATURE_EXTRACTION)
        self.llm = get_peft_model(self.llm, llm_lora)

        # 3. Adapters & Learnable Temperature
        self.image_adapter = ImageAdapter(clip_dim, llm_dim).to(torch.bfloat16)
        self.projection = ProjectionHead(llm_dim, embed_dim).to(torch.bfloat16)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #Adapter for Ref
    def encode_target(self, pixel_values):
        """z_i: 768-dim retrieval space, used as contrastive target (no grad needed)"""
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return F.normalize(image_embeds, dim=-1)   # [B, 768]

    def encode_reference(self, pixel_values):
        """h'_i: 1024-dim pooler output, feeds into ImageAdapter → LLM (needs grad)"""
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return F.normalize(outputs.pooler_output, dim=-1)   # [B, 1024]

    def encode_query(self, visual_token=None, texts=None, device=None):
        B = visual_token.shape[0] if visual_token is not None else len(texts)
        instructions = []

        # Accurately implement paper's instruction format
        for i in range(B):
            prompt = "Instruct: Find the image that matches the query.\nQuery:\n"
            if visual_token is not None:
                prompt += f"Image: {self.image_token}\n"
            if texts is not None:
                prompt += f"Text: {texts[i]}"
            instructions.append(prompt.strip())

        encoded = self.tokenizer(instructions, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        # Robustly get embedding table through PEFT
        inputs_embeds = self.llm.get_input_embeddings()(encoded["input_ids"])

        # Inject visual token precisely where <image> is
        if visual_token is not None:
            mask = (encoded["input_ids"] == self.image_token_id)
            vt = visual_token.squeeze(1).to(inputs_embeds.dtype)
            #Changed:
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = vt

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=encoded["attention_mask"])
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)

    def forward_queries(self, h_star, captions, mod_texts, device):
        """Processes the 3 query variants once h_star is computed externally."""
        visual_token = self.image_adapter(h_star.to(torch.bfloat16))
        
        c_v = self.encode_query(visual_token=visual_token, texts=None, device=device)       # Eq 3
        c_w = self.encode_query(visual_token=None, texts=captions, device=device)           # Eq 4
        c   = self.encode_query(visual_token=visual_token, texts=mod_texts, device=device)  # Eq 5
        
        return c_v, c_w, c