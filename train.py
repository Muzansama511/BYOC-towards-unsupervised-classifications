
import os
import copy
import random
 
import numpy as np
import torch
 
import lightning as L
# from lightning.pytorch.loggers import LitLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
# ── your own modules — adjust these imports to match your actual filenames ──
from Dataset import get_dataloader, get_tiktokenizer
from Model.model import RoutedBYOCLightning
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler



def main():
    # ── config (swap this block for Hydra later if you want sweeps) ──
    num_workers = 4
    max_steps = -1
    val_check_interval = 10
    lr = 1e-4
    shuffle=False
    precision = "32"
    accumulate_grad_batches = 1
    warmup = 500
    hidden = 64
    num_layers = 4
    num_decoders = 10
    do_switch = True
    switch_every = 1000
    use_deepnet_init = False  # Set to True if you want to use DeepNet initialization
    
    # ── tokenizer / special tokens ──
    enc, meta = get_tiktokenizer()
    cls_id = meta["CLS_ID"]
    sep_id = meta["SEP_ID"]
    mask_id = meta["MASK_ID"]
    pad_id = meta["PAD_ID"]
    vocab_size = enc.n_vocab
 
    # ── data ──
    train_loader = get_dataloader(train_or_val="train", batch_size=8, num_workers=num_workers, max_length=128, perc_to_use=0.01, shuffle=shuffle)
    val_loader = get_dataloader(train_or_val="val", batch_size=8, num_workers=num_workers, max_length=128, perc_to_use=0.0001, shuffle=False)
 
    # ── model ──
    lit_model = RoutedBYOCLightning(vocab_size=vocab_size, 
                                    hidden_dim=hidden,
                                    output_dim=hidden,
                                    num_decoders=num_decoders,
                                    num_layers = num_layers,
                                    do_switch = do_switch,
                                    switch_every=switch_every,
                                    warmup_steps=warmup,
                                    lr=lr,
                                    use_deepnet_init=use_deepnet_init,
                                    # ema_update_every=ema_update_every
                                    )
    
    # # ── logging ──
    # wandb_logger = WandbLogger(
    #     project="world-model-encoder",
    #     name="run-cls-mlm-v1",
    #     log_model=True,
    # )
 
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="step{step}-valloss{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=val_check_interval,
    )
 
    logger = TensorBoardLogger(
    save_dir="logs/",
    name="jea"
    )
 
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # ── trainer ──
    trainer = L.Trainer(
        # max_epochs=1,
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        precision=precision,
        # accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=100,
        # limit_val_batches=0.0,
        val_check_interval=val_check_interval,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        profiler=SimpleProfiler(dirpath="./profiler_logs", filename="simple_profile"),
        # profiler=AdvancedProfiler(dirpath="./profiler_logs", filename="advanced_profile"),
        # profiler=PyTorchProfiler(dirpath="./profiler_logs", filename="pt_profile_single_optim_change", export_to_chrome=True),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir="checkpoints/",
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()