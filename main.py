import pandas as pd
from torch.utils.data import Dataset, DataLoader
from vae import EmbedVAE  # Make sure the EmbedVAE class is defined in vae.py

# Custom dataset to handle DNA methylation and gene expression data
class OmicsDataset(Dataset):
    def __init__(self, dna_methylation, gene_expression):
        self.dna_methylation = dna_methylation
        self.gene_expression = gene_expression
    
    def __len__(self):
        return len(self.dna_methylation)
    
    def __getitem__(self, idx):
        dna_methylation_sample = self.dna_methylation.iloc[idx].values.astype(np.float32)
        gene_expression_sample = self.gene_expression.iloc[idx].values.astype(np.float32)
        return dna_methylation_sample, gene_expression_sample

# Function to create a dataloader for the omics data
def create_omics_dataloader(dna_methylation_path, gene_expression_path, batch_size=32):
    dna_methylation = pd.read_csv(dna_methylation_path, index_col=0)
    gene_expression = pd.read_csv(gene_expression_path, index_col=0)
    
    dataset = OmicsDataset(dna_methylation, gene_expression)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Assuming the use of a CUDA-capable device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the data
data_dir = ""
dna_methylation_path = 'preprocessed/filtered_DNA_methylation27.csv'
gene_expression_path = 'preprocessed/filtered_rna_seq.csv'

# Create the dataloader
omics_dataloader = create_omics_dataloader(os.path.join(data_dir,dna_methylation_path), os.path.join(data_dir, gene_expression_path))

# Initialize the VAE model
vae_model = EmbedVAE().to(device)

# Initialize the NetManager
net_manager = NetManager(
    model=vae_model,
    device=device,
    train_loader=omics_dataloader,  # Using the same dataloader for both training and testing for simplicity
    test_loader=omics_dataloader
)

# Set a writer for logging
net_manager.set_writer('vae_training')

# Train the model for a specified number of epochs
epochs = 100  # Adjust the number of epochs as needed
net_manager.train(epochs)

# After training, you can save the model's state
net_manager.save_net('vae_model_state.pth')

# Additional functionality such as plotting latent space can be called as needed
# net_manager.plot_latent_space()

# Note that this implementation does not handle separate validation/testing datasets
# or early stopping. These would be necessary for a full training pipeline.
