import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm

class NetManager:
    """
    Net manager for the VAE
    """

    def __init__(
            self,
            model,
            device,
            train_loader=None,
            test_loader=None,
            lr=1e-3):
        """
        Constructor
        """
        self.model = model.to(device)
        self.device = device
        self.writer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def set_writer(self, board_name):
        """
        Sets a torch writer object. The logs will be generated in logs/name
        """
        if isinstance(self.writer, SummaryWriter):
            self.writer.close()

        if board_name is None:
            self.writer = None
        else:
            self.writer = SummaryWriter("logs/" + board_name)

    def train(self, epochs, log_interval=10):
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (dna_methylation, gene_expression) in enumerate(self.train_loader):
                dna_methylation = dna_methylation.to(self.device)
                gene_expression = gene_expression.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(dna_methylation, gene_expression)

                # Calculate loss
                loss = self.loss_function(recon_batch, dna_methylation, gene_expression, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                if batch_idx % log_interval == 0 and self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss / log_interval, epoch * len(self.train_loader) + batch_idx)
                    train_loss = 0

            print(f'Epoch {epoch}, Loss: {train_loss / len(self.train_loader)}')

            if self.writer is not None:
                self.writer.flush()

    def loss_function(self, recon_x, dna_methylation, gene_expression, mu, logvar):
        bce_dna = F.binary_cross_entropy(recon_x[:, :dna_methylation.size(1)], dna_methylation, reduction='sum')
        bce_gene = F.binary_cross_entropy(recon_x[:, dna_methylation.size(1):], gene_expression, reduction='sum')
        bce = bce_dna + bce_gene
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld
