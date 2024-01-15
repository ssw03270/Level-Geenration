import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from datetime import datetime

import numpy as np
import random
from tqdm import tqdm

from model import Transformer
from dataloader import CraftAssistDataset

import wandb

class Trainer:
    def __init__(self, d_model, d_hidden, n_head, n_layer,
                 batch_size, max_epoch,
                 dropout, use_checkpoint, checkpoint_epoch, use_wandb,
                 val_epoch, save_epoch, lr, save_dir_path):
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
        self.lr = lr

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Only the first dataset initialization will load the full dataset from disk
        self.train_dataset = CraftAssistDataset(data_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.transformer = Transformer(d_model, d_hidden, n_head, n_layer, dropout).to(device=self.device)

        # optimizer
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False, no_deprecation_warning=True)

        # scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_train_steps)

    def cross_entropy_loss(self, pred, trg, mask):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed BCE loss.
        """
        trg = torch.nn.functional.one_hot(trg, pred.shape[2])
        pred = pred.view(-1, pred.size(-1))
        trg = trg.view(-1, trg.size(-1))
        mask = mask.view(-1)

        loss = F.cross_entropy(pred, trg.float(), reduction='none')
        masked_loss = loss * mask.float()

        return masked_loss.sum() / mask.float().sum()

    def get_accuracy(self, pred, trg, mask):
        # 가장 높은 확률을 갖는 클래스 인덱스를 얻음
        pred = torch.argmax(pred, dim=-1)
        # 정확도 계산
        correct = (trg[mask] == pred[mask]).sum().item()
        total = mask.sum().item()

        return correct, total

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        if self.use_wandb:
            wandb.watch(self.transformer, log='all')

        for epoch in range(epoch_start, self.max_epoch):
            loss_parent_sum = 0
            loss_dir_sum = 0
            true_parent_sums = 0
            true_dir_sums = 0
            problem_parent_sums = 0
            problem_dir_sums = 0

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                position_sequence, block_id_sequence, block_semantic_sequence, \
                    dir_sequence, parent_sequence, pad_mask_sequence, terrain_mask_sequence = data
                position_sequence = position_sequence.to(device=self.device)
                block_id_sequence = block_id_sequence.to(device=self.device)
                block_semantic_sequence = block_semantic_sequence.to(device=self.device)
                dir_sequence = dir_sequence.to(device=self.device)
                parent_sequence = parent_sequence.to(device=self.device)
                pad_mask_sequence = pad_mask_sequence.to(device=self.device)
                terrain_mask_sequence = terrain_mask_sequence.to(device=self.device)

                # Get the model's predictions
                parent_output, dir_output = self.transformer(position_sequence, block_id_sequence,
                                                             block_semantic_sequence, pad_mask_sequence)

                # Compute the losses
                mask = pad_mask_sequence & terrain_mask_sequence
                loss_parent = self.cross_entropy_loss(parent_output, parent_sequence.detach(), mask.detach())
                loss_dir = self.cross_entropy_loss(dir_output, dir_sequence.detach(), mask.detach())
                loss = loss_parent + loss_dir
                # Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                loss_parent_sum += loss_parent.detach()
                loss_dir_sum += loss_dir.detach()
                true_parent_sum, problem_parent_sum = self.get_accuracy(parent_output.detach(), parent_sequence.detach(), mask.detach())
                true_parent_sums += true_parent_sum
                problem_parent_sums += problem_parent_sum
                true_dir_sum, problem_dir_sum = self.get_accuracy(dir_output.detach(), dir_sequence.detach(), mask.detach())
                true_dir_sums += true_dir_sum
                problem_dir_sums += problem_dir_sum

                # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
            loss_parent_mean = loss_parent_sum / (len(self.train_dataloader))
            loss_dir_mean = loss_dir_sum / (len(self.train_dataloader))
            true_parent_mean = true_parent_sums / problem_parent_sums
            true_dir_mean = true_dir_sums / problem_dir_sums
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE parent: {loss_parent_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE dir: {loss_dir_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy parent: {true_parent_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy dir: {true_dir_mean:.4f}")

            if self.use_wandb:
                wandb.log({"Train ce parent": loss_parent_mean}, step=epoch + 1)
                wandb.log({"Train ce dir": loss_dir_mean}, step=epoch + 1)
                wandb.log({"Train accuracy parent": true_parent_mean}, step=epoch + 1)
                wandb.log({"Train accuracy dir": true_dir_mean}, step=epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                # 체크포인트 데이터 준비
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.transformer.state_dict(),
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
    parser.add_argument("--d_model", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--d_hidden", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--n_head", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=6, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="transformer", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"CraftAssist_{current_time}"

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

    # Create a Trainer instance and start the training process
    trainer = Trainer(d_model=opt.d_model, d_hidden=opt.d_hidden, n_head=opt.n_head, n_layer=opt.n_layer,
                      batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      use_wandb=opt.use_wandb, dropout=opt.dropout, use_checkpoint=opt.use_checkpoint,
                      checkpoint_epoch=opt.checkpoint_epoch, val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      lr=opt.lr, save_dir_path=opt.save_dir_path)

    trainer.train()

    wandb.finish()