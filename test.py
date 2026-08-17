from Model.model import RoutedBYOLLightning
import torch
from Dataset import get_dataloader, get_tiktokenizer

def main():
    # ── config (swap this block for Hydra later if you want sweeps) ──
    num_workers = 4
    max_steps = -1
    val_check_interval = 100
    lr = 1e-4
    shuffle=False
    ema_update_every = 5
    ema_momentum = 0.999
    precision = "32"
    accumulate_grad_batches = 1
    warmup = 1000
    hidden = 64
    num_layers = 4
    num_teachers = 4
    use_deepnet_init = False  # Set to True if you want to use DeepNet initialization
    
    # ── tokenizer / special tokens ──
    enc, meta = get_tiktokenizer()
    cls_id = meta["CLS_ID"]
    sep_id = meta["SEP_ID"]
    mask_id = meta["MASK_ID"]
    pad_id = meta["PAD_ID"]
    vocab_size = enc.n_vocab
 
    # ── data ──
    train_loader = get_dataloader(train_or_val="train", batch_size=32, num_workers=num_workers, max_length=128, perc_to_use=0.01, shuffle=shuffle)
    val_loader = get_dataloader(train_or_val="val", batch_size=32, num_workers=num_workers, max_length=128, perc_to_use=0.0001, shuffle=False)
 
    # ── model ──
    # lit_model = RoutedBYOLLightning(vocab_size=vocab_size, 
    #                                 hidden_dim=hidden,
    #                                 output_dim=hidden,
    #                                 num_teachers=num_teachers,
    #                                 num_layers = num_layers,
    #                                 ema_momentum=ema_momentum,
    #                                 lr=lr,
    #                                 warmup_steps=warmup,
    #                                 use_deepnet_init=use_deepnet_init,
    #                                 accumulate_grad_steps=accumulate_grad_batches, # custom hparam
    #                                 # ema_update_every=ema_update_every
    #                                 )
    
    lit_model = RoutedBYOLLightning.load_from_checkpoint("checkpoints/stepstep=70000-vallossval_loss=0.224.ckpt")
    
    for batch in val_loader:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for k,v in batch.items():
            batch[k] = batch[k].to(device)
        loss, stats = lit_model.test_run_(batch)
        print(stats["tokens_online"])

if __name__ == "__main__":
    main()