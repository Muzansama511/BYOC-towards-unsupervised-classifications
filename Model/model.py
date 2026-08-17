from types import NoneType

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchview import draw_graph
from torch.optim.lr_scheduler import LambdaLR
import math
from time import sleep, time
# from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss

try:
    from .BERTmodel import *
except:
    from BERTmodel import *



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=12, use_deepnet_init: bool = True):
        
        super().__init__()

        self.encoder = BERT(
            vocab_size=input_dim,
            hidden=hidden_dim,
            n_layers=num_layers,
            attn_heads=2,
            dropout=0.1,
            base=256,
            use_deepnet_init=use_deepnet_init,
            encoder=True
        )


    def forward(self, x):
        h = self.encoder(x)
        return h

class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=12, use_deepnet_init: bool = True, accumulate_grad_steps=1, lr=1e-4):
        super().__init__()

        # self.decoder = BERT(
        #     vocab_size=input_dim,
        #     hidden=hidden_dim,
        #     n_layers=num_layers,
        #     attn_heads=2,
        #     dropout=0.1,
        #     base=256,
        #     use_deepnet_init=use_deepnet_init,
        #     encoder=False
        # )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        h = self.decoder(x)
        return h
    
    
class DecoderRouter(nn.Module):
    """Routes each sample to one decoder using straight-through top-1 gating."""

    def __init__(
        self,
        hidden_dim,
        num_decoders,
        lb_weight=0.01,
        max_lb_weight=0.2,
        decay=0.99,
    ):
        super().__init__()

        self.gate = nn.Linear(hidden_dim, num_decoders)
        self.num_decoders = num_decoders
        self.lb_weight = lb_weight
        self.max_lb_weight = max_lb_weight
        self.decay = decay

        self.register_buffer(
            "running_usage",
            torch.full((num_decoders,), 1 / num_decoders),
        )

    def forward(self, cls):
        probs = F.softmax(self.gate(cls), dim=-1)
        top1 = probs.argmax(dim=-1)

        hard = F.one_hot(top1, self.num_decoders).float()
        gate = (hard - probs).detach() + probs

        importance = probs.mean(0)
        load = hard.mean(0)
        lb_loss = self.num_decoders * (importance * load).sum()

        with torch.no_grad():
            self.running_usage.mul_(self.decay).add_(
                importance, alpha=1 - self.decay
            )

            usage = self.running_usage.clamp_min(1e-8)
            entropy = -(usage * usage.log()).sum()
            max_entropy = torch.log(
                torch.tensor(self.num_decoders, device=usage.device)
            )
            health = (entropy / max_entropy).item()

            weight = self.lb_weight + (1 - health) * (
                self.max_lb_weight - self.lb_weight
            )

        return top1, gate, lb_loss, weight, health

class FusedDecoders(nn.Module):
    def __init__(self, output_size, hidden_dim,  num_decoders=4):
        super().__init__()
        self.num_decoders = num_decoders
        self.w1 = nn.Parameter(torch.empty(num_decoders, hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_decoders, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_decoders, hidden_dim, output_size))
        self.b2 = nn.Parameter(torch.zeros(num_decoders, output_size))
        
        nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)
        
    def forward(self, x, decoder_ids):
        
        h1 = torch.bmm(x, self.w1[decoder_ids]) + self.b1[decoder_ids].unsqueeze(1)      
        h1 = F.gelu(h1)
        out = torch.bmm(h1, self.w2[decoder_ids]) + self.b2[decoder_ids].unsqueeze(1) 
        
        # # printt(out.shape)
        return out
        
class RoutedBYOC(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        output_dim,
        num_decoders,
        num_layers=12,
        use_deepnet_init: bool = True,
        lr=1e-4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_decoders = num_decoders
        self.encoder = Encoder(
            vocab_size, hidden_dim, num_layers, use_deepnet_init=use_deepnet_init
        )

        self.decoders = FusedDecoders(vocab_size, hidden_dim, num_decoders)
        
        # self.decoders = nn.ModuleList([
        #     Decoder(
        #         vocab_size, hidden_dim, num_layers, use_deepnet_init=use_deepnet_init, lr=lr,
        #     )
            
        #     for _ in range(num_decoders)
        # ])
        
        self.router = DecoderRouter(hidden_dim, num_decoders)
        self.CELoss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x_online, x_decoder, mask_positions):
        s = time()
        y_online = self.encoder(x_online)
        # printt(f"#### Step : Encoder forward pass took {time() - s:.4f} seconds")
        cls = y_online[...,0,:]
        s = time()
        top1, _, lb_loss, lb_weight, health = self.router(cls)
        # printt(f"#### Step : Router forward pass took {time() - s:.4f} seconds")
        
        s = time()
        # y_decoder = torch.zeros(list(x_decoder.shape) + [self.vocab_size], device=x_decoder.device)
        # for tid in top1.unique():
        #     idx = (top1 == tid).nonzero(as_tuple=True)[0]
        #     decoder_output = self.decoders[tid.item()](y_online[idx])
        #     y_decoder[idx] = decoder_output
        
        y_decoder = self.decoders(y_online, top1)
        # printt(f"#### Step : Decoder forward pass took {time() - s:.4f} seconds")
        
        s = time()
        mask = mask_positions.float()
        mask_flat = mask_positions.bool().view(-1)
        # token_pred = torch.masked_select(y_decoder, mask.bool().unsqueeze(-1)).view(-1, self.vocab_size)
        # decoder_tokens = torch.masked_select(x_decoder, mask.bool()).view(-1)
        # token_pred = y_decoder.view(-1, self.vocab_size)[mask_flat]
        token_pred = y_decoder.view(-1, self.vocab_size)*mask_flat.unsqueeze(-1)
        decoder_tokens = x_decoder.view(-1)*mask_flat
        # printt(f"#### Step : Masking took {time() - s:.4f} seconds")
        
        s = time()
        token_loss = self.CELoss(token_pred, decoder_tokens)
        # token_loss = self.CELoss(token_pred, decoder_tokens)*(mask_flat.numel()/mask_flat.sum())
        # printt(f"#### Step : Loss computation took {time() - s:.4f} seconds")
        # token_loss = ((token_pred - decoder_tokens.detach())**2).mean() # L2 loss

        # total_loss = token_loss + cls_loss + lb_weight * lb_loss

        total_loss = token_loss + (lb_weight * lb_loss)
        s = time()
        stats = {}
        # stats = {
        #     "token_loss": token_loss.detach(),
        #     "lb_loss": lb_loss.detach(),
        #     "lb_weight": lb_weight,
        #     "routing_health": health,
        #     "num_decoders_used": top1.unique().numel(),
        #     "routing": top1,
        # }
        # printt(f"#### Step : stats took {time() - s:.4f} seconds")
        return total_loss, stats


class RoutedBYOCLightning(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        output_dim,
        num_decoders,
        num_layers=12,
        do_switch=True,
        switch_every=100,
        warmup_steps=10000,
        lr=1e-4,
        weight_decay=0.01,
        use_deepnet_init: bool = True,
    ):
        super().__init__()    
        self.save_hyperparameters()
        self.automatic_optimization = False
        # self.first = True
        self.in_warmup = True
        self.switch = 'encoder_decoder'
        self.model = (RoutedBYOC(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_decoders=num_decoders,
            num_layers=num_layers,
            use_deepnet_init=use_deepnet_init,
            lr=lr,
        ))

    def forward(self, x_online, x_decoder, mask_positions):
        return self.model(x_online, x_decoder, mask_positions)

    def training_step(self, batch, batch_idx):
        # print(self.hparams.do_switch, self.in_warmup, (self.global_step + 1) , self.hparams.switch_every)
        if self.hparams.do_switch and (not self.in_warmup) and (self.global_step + 1) % self.hparams.switch_every == 0:
            self.switch = 'decoder' if self.switch == 'encoder' else 'encoder'
            # printt(f"Switching to {self.switch} optimization at step {self.global_step + 1}")
        
        optimizer = self.optimizers()
        schedulers = self.lr_schedulers()
        # encoder_optim, router_optim, *decoder_optimizers = optimizer
        encoder_optim, router_optim, decoder_optimizers = optimizer
        encoder_sched, router_sched, decoder_sched = schedulers
        x_online, x_decoder, mask_positions = batch["input"], batch["output"], batch["consider_for_loss"]
        s = time()
        loss, stats = self.model.forward(
            x_online,
            x_decoder,
            mask_positions,
        )
        # printt(f"Step {self.global_step + 1}: Forward pass took {time() - s:.4f} seconds")

        
        encoder_optim.zero_grad()
        router_optim.zero_grad()
        s = time()
        self.manual_backward(loss)
        # printt(f"Step {self.global_step + 1}: Backward pass took {time() - s:.4f} seconds")
        
        s = time()
        encoder_optim.step()
        # printt(f"Step {self.global_step + 1}: Encoder optimizer step took {time() - s:.4f} seconds")
        
        s = time()
        if 'encoder' in self.switch:
            router_optim.step()
        # printt(f"Step {self.global_step + 1}: Router optimizer step took {time() - s:.4f} seconds")
        
        s = time()
        if 'decoder' in self.switch:
            # for tid in stats["routing"].unique():
            #     decoder_optimizers[tid].step()
            #     decoder_optimizers[tid].zero_grad()
            decoder_optimizers.step()
            decoder_optimizers.zero_grad()
        # printt(f"Step {self.global_step + 1}: Decoder optimizer step took {time() - s:.4f} seconds")
        
        # schedules
            
        encoder_sched.step()
        router_sched.step()
        decoder_sched.step()
        
        s = time()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/lr", encoder_sched.get_last_lr()[0], sync_dist=True)
        log_code = 0
        if 'encoder' in self.switch:
            log_code+=1
        if 'decoder' in self.switch:
            log_code+=2
        self.log("train/mode" , log_code, prog_bar=True, sync_dist=True)
        # self.log("train/token_loss", stats["token_loss"], sync_dist=True)
        # self.log("train/lb_loss", stats["lb_loss"], sync_dist=True)
        # self.log("train/lb_weight", stats["lb_weight"], sync_dist=True)
        # self.log("train/routing_health", stats["routing_health"], sync_dist=True)
        # self.log(
        #     "train/decoders_used",
        #     stats["num_decoders_used"],
        #     sync_dist=True,
        # )
        # printt(f"Step {self.global_step + 1}: Logging took {time() - s:.4f} seconds")
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.encoder.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        router_optimizer = torch.optim.AdamW(self.model.router.parameters(),
                                             lr=self.hparams.lr, 
                                             weight_decay=self.hparams.weight_decay)
        # decoder_optimizers = [torch.optim.AdamW(
        #     d.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        # ) for d in self.model.decoders]
        decoder_optimizers = torch.optim.AdamW(self.model.decoders.parameters(), 
                                               lr = self.hparams.lr, 
                                               weight_decay=self.hparams.weight_decay)
        
        def lr_lambda(step):
            warmup = self.hparams.warmup_steps
            total = self.trainer.estimated_stepping_batches
            if step < warmup:
                # linear warmup
                self.in_warmup = True
                return (step + 1) / max(1, warmup)
            # cosine decay after warmup
            self.in_warmup = False
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        # def lr_const_placeholder(step):
        #     return 1.0 
                
        scheduler_encoder = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_router = LambdaLR(router_optimizer, lr_lambda=lr_lambda)
        scheduler_decoder = LambdaLR(decoder_optimizers, lr_lambda=lr_lambda)
        # scheduler = LambdaLR(optimizer, lr_lambda=lr_const_placeholder)  # Placeholder scheduler to avoid Lightning's warning
        

        return ([optimizer, router_optimizer, decoder_optimizers],
                [{"scheduler": scheduler_encoder, "interval": "step", "frequency": 1},
                 {"scheduler": scheduler_router, "interval": "step", "frequency": 1},
                 {"scheduler": scheduler_decoder, "interval": "step", "frequency": 1}])

