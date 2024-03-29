import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch.distributed as dist
from datetime import datetime
import torch.nn as nn

from transformers import BertTokenizer

import numpy as np
import random
from tqdm import tqdm

from model import CategoryTransformer
from dataloader import CraftAssistDataset

import wandb


def cross_entropy_loss(pred, trg, mask):
    """
    Compute the binary cross-entropy loss between predictions and targets.

    Args:
    - pred (torch.Tensor): Model predictions.
    - trg (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Computed BCE loss.
    """
    pred = pred.reshape(-1, pred.size(-1))
    trg = trg.reshape(-1)
    mask = mask.reshape(-1)

    loss = F.cross_entropy(pred, trg, reduction='none')
    masked_loss = loss * mask.float()

    return masked_loss.sum() / mask.float().sum()


def get_accuracy(pred, trg, mask):
    # 가장 높은 확률을 갖는 클래스 인덱스를 얻음
    pred = torch.argmax(pred, dim=-1)
    # 정확도 계산
    correct = (trg[mask] == pred[mask]).sum().item()
    total = mask.sum().item()

    return correct, total


class Trainer:
    def __init__(self, d_model, d_hidden, n_head, n_layer,
                 batch_size, max_epoch,
                 dropout, use_checkpoint, checkpoint_epoch, use_wandb,
                 val_epoch, save_epoch, lr, save_dir_path, local_rank):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of transformer layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_wandb = use_wandb
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.save_dir_path = save_dir_path
        self.lr = lr * torch.distributed.get_world_size()
        self.local_rank = local_rank

        # Set the device for training
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Dataset and Dataloader
        self.train_dataset = CraftAssistDataset(data_type='train')
        self.train_sampler = DistributedSampler(dataset=self.train_dataset,
                                                num_replicas=torch.distributed.get_world_size(), rank=self.local_rank,
                                                shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                                           num_workers=8, pin_memory=True)

        # Initialize the Transformer model
        self.transformer = CategoryTransformer(d_model, d_hidden, n_head, n_layer, dropout).to(self.device)
        self.transformer = nn.parallel.DistributedDataParallel(self.transformer, device_ids=[self.local_rank],
                                                               find_unused_parameters=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Optimizer and Scheduler
        param_optimizer = list(self.transformer.module.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False,
                               no_deprecation_warning=True)

        # Scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_train_steps)

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        # if self.use_wandb:
        #     if self.local_rank == 0:
        #         wandb.watch(self.transformer.module, log='all')

        for epoch in range(epoch_start, self.max_epoch):
            loss_category_sum = torch.Tensor([0.0]).to(self.device)
            true_category_sums = torch.Tensor([0.0]).to(self.device)
            problem_category_sums = torch.Tensor([0.0]).to(self.device)

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                position_sequence, id_sequence, category_sequence, next_category_sequence, \
                    pad_mask_sequence, terrain_mask_sequence, text_sequence = data

                text_sequence = self.tokenizer(text_sequence, padding=True, truncation=True, return_tensors="pt")
                text_sequence = text_sequence.to(device=self.device)

                position_sequence = position_sequence.to(device=self.device)
                id_sequence = id_sequence.to(device=self.device)
                category_sequence = category_sequence.to(device=self.device)
                next_category_sequence = next_category_sequence.to(device=self.device)
                pad_mask_sequence = pad_mask_sequence.to(device=self.device)
                terrain_mask_sequence = terrain_mask_sequence.to(device=self.device)

                # Get the model's predictions
                category_output = self.transformer(text_sequence,
                                                   position_sequence,
                                                   id_sequence,
                                                   category_sequence,
                                                   pad_mask_sequence)

                # Compute the losses
                mask = pad_mask_sequence & terrain_mask_sequence
                loss_category = cross_entropy_loss(category_output, next_category_sequence.detach(), mask.detach())
                loss = loss_category

                # Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                dist.all_reduce(loss_category, op=dist.ReduceOp.SUM)
                loss_category_sum += loss_category.detach()

                true_category_sum, problem_category_sum = get_accuracy(category_output.detach(),
                                                                       next_category_sequence.detach(), mask.detach())
                dist.all_reduce(torch.tensor(true_category_sum).to(self.device), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(problem_category_sum).to(self.device), op=dist.ReduceOp.SUM)
                true_category_sums += true_category_sum
                problem_category_sums += problem_category_sum

            if self.local_rank == 0:
                loss_category_mean = loss_category_sum.item() / (len(self.train_dataloader) * dist.get_world_size())
                true_category_mean = true_category_sums.item() / (problem_category_sums.item())

                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE category: {loss_category_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy category: {true_category_mean:.4f}")

                if self.use_wandb:
                    wandb.log({"Train ce category": loss_category_mean}, step=epoch + 1)
                    wandb.log({"Train accuracy category": true_category_mean}, step=epoch + 1)

                if (epoch + 1) % self.save_epoch == 0:
                    # 체크포인트 데이터 준비
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.transformer.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }

                    save_path = os.path.join("./models", self.save_dir_path)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "transformer_epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--d_hidden", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--n_head", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=3, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="transformer", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--local-rank", type=int)

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"Category_{current_time}"

    if opt.local_rank == 0:
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(project='level-generation', entity='ssw03270', config=vars(opt), name=opt.save_dir_path)

        for key, value in wandb.config.items():
            setattr(opt, key, value)

        save_path = os.path.join("./models", opt.save_dir_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        config_file_path = os.path.join(save_path, "config.txt")
        with open(config_file_path, "w") as f:
            for arg in vars(opt):
                f.write(f"{arg}: {getattr(opt, arg)}\n")

        for arg in vars(opt):
            print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    rank = opt.local_rank
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') == "cuda:0":
            dist.init_process_group("gloo")

        else:
            dist.init_process_group('nccl')

    # Create a Trainer instance and start the training process
    trainer = Trainer(d_model=opt.d_model, d_hidden=opt.d_hidden, n_head=opt.n_head, n_layer=opt.n_layer,
                      batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      use_wandb=opt.use_wandb, dropout=opt.dropout, use_checkpoint=opt.use_checkpoint,
                      checkpoint_epoch=opt.checkpoint_epoch, val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      lr=opt.lr, save_dir_path=opt.save_dir_path, local_rank=opt.local_rank)

    trainer.train()

    if opt.local_rank == 0:
        wandb.finish()
