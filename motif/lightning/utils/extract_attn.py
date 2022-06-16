import sys

from transformers.models.auto import AutoModelForSequenceClassification
sys.path.insert(0, "../")
from src.modules.finetune_model import FineTuneModule
from src.datamodules.FineTuneDM import FineTuneDM
from torch.utils.data import ConcatDataset, DataLoader
import torch
from datasets import concatenate_datasets, load_from_disk
import pandas as pd


device = torch.device("cuda")

# arch, seq_len, ckpt = "allenai/longformer-base-4096", 4096, "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/79"
arch, seq_len, ckpt = "roberta-large", 512, "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/79"
dm = FineTuneDM(
    "/home/jiashu/uncanny_valley/datasets", "80_20_0", 
    None, 15, 10, True, arch, seq_len, "/home/jiashu/uncanny_valley/datasets/cache"
)
dm.setup("fit")

ds = concatenate_datasets([
    dm.train_dataloader().dataset,
    dm.val_dataloader().dataset,
])
dataloader = DataLoader(ds, batch_size=4, num_workers=5, pin_memory=True, shuffle=False)

# transformer = (
#     AutoModelForSequenceClassification.from_pretrained(ckpt)
#     .base_model
#     .to(device)
#     .eval()
# )

transformer = (
    FineTuneModule.load_from_checkpoint("/home/jiashu/uncanny_valley/motif/lightning/logs/ckpt/35/epoch_029-loss_1.161-acc_0.800.ckpt")
        .transformer
).to(device).eval()
to_save = []
with torch.no_grad():
    for data in dataloader:
        output = transformer(data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            output_attentions=True)
        # tuple of all layers 
        # each (batch_size, num_heads, sequence_length, sequence_length)
        attns = output.attentions
        # take [CLS]'s attn over all rest
        # (bsz, seq_len - 1)
        first_layer = attns[0].mean(1)[:, 0, 1:data["input_ids"].size(1)].cpu().numpy()
        # (bsz, seq_len - 1)
        avg = torch.stack([
                attn.mean(1)[:, 0, 1:data["input_ids"].size(1)]
                for attn in attns
            ]).mean(0).cpu().numpy()
        for atu, desc, text, input_ids, f, a in zip(data["atu"], data["desc"], data["text"], data["input_ids"].cpu().numpy(), first_layer, avg): 
            to_save.append([atu, desc, text, input_ids, f, a])

df = pd.DataFrame(to_save, columns=["atu", "desc", "text", "tokens", "first_layer", "avg"])
# df.to_pickle("./hug_attn.pkl")
df.to_pickle("./lig_attn.pkl")

