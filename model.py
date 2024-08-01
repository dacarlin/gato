from random import seed
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv, global_add_pool
from Bio.Data import IUPACData


# Set seed for reproducibility 
torch.manual_seed(42)
seed(42)


# Constants
NUM_AMINO_ACIDS = 20
HEADS = 4 


# Global map from residues to tokens 
aa_to_idx = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
        # add noncanonicals 
        "DM0": 11, 
        "MLY": 11, 
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

# Reverse map of tokens to amino acids 
idx_to_aa = {v: k for k, v in aa_to_idx.items()}
idx_to_aa.update({11: "LYS"})



###############################################################################
# Main model implementation                                                   #
###############################################################################


class ProteinLigandGNN(nn.Module):
    """
    Graph Neural Network for protein sequence design, incorporating ligand information.

    This model processes both protein backbone features and ligand atom features
    to predict amino acid sequences that are compatible with the given structure
    and ligand binding.

    Args:
        protein_features (int): Number of input features for protein nodes.
        ligand_features (int): Number of input features for ligand nodes.
        edge_features (int): Number of input features for edges.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of GNN layers.
    """

    def __init__(
        self, protein_features, ligand_features, edge_features, hidden_dim, num_layers
    ):
        super(ProteinLigandGNN, self).__init__()
        self.protein_embedding = nn.Linear(protein_features, hidden_dim)
        self.ligand_embedding = nn.Linear(ligand_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        # add positional embeddings 

        # Multiple GAT layers for message passing
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, hidden_dim, heads=HEADS, concat=False, edge_dim=hidden_dim)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Final output layer for amino acid prediction
        self.output_layer = nn.Linear(hidden_dim * 2, NUM_AMINO_ACIDS)

    def forward(self, x, edge_index, edge_attr, batch, ligand_x, ligand_batch):
        """
        Forward pass of the ProteinLigandGNN.

        Args:
            x (Tensor): Protein node features.
            edge_index (Tensor): Graph connectivity in COO format.
            edge_attr (Tensor): Edge features.
            batch (Tensor): Batch vector for protein nodes.
            ligand_x (Tensor): Ligand node features.
            ligand_batch (Tensor): Batch vector for ligand nodes.

        Returns:
            Tensor: Logits for amino acid prediction at each protein node.
        """
        protein_x = self.protein_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        # add positional embeddings 

        # Handle the case where we input only a single sequence, as in generating
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if ligand_x.size(0) > 0:
            ligand_x = self.ligand_embedding(ligand_x)
            # Combine protein and ligand features
            combined_x = torch.cat([protein_x, ligand_x], dim=0)
            combined_batch = torch.cat([batch, ligand_batch], dim=0)
        else:
            combined_x = protein_x
            combined_batch = batch

        # Apply GAT layers with residual connections and layer normalization
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            x_res = combined_x
            combined_x = gat(combined_x, edge_index, edge_attr)
            combined_x = ln(combined_x + x_res)
            combined_x = F.relu(combined_x)

        # Global pooling of ligand features into a single "residue" at the end 
        if ligand_x.size(0) > 0:
            ligand_pooled = global_add_pool(ligand_x, ligand_batch)
        else:
            ligand_pooled = torch.zeros(
                (1, self.protein_embedding.out_features), device=protein_x.device
            )

        # Concatenate protein features with pooled ligand features, in the sequence dimension, 
        # so the ligand is a fixed-sized representation like a residue is 
        protein_x = combined_x[: protein_x.size(0)]
        output = torch.cat([protein_x, ligand_pooled[batch]], dim=1)

        return self.output_layer(output)




###############################################################################
# Training functions                                                          #
###############################################################################


def train(model, train_loader, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (ProteinLigandGNN): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to use for training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    
    for data in train_loader:
        t0 = time.time()
        #print(data)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.ligand_x, data.ligand_batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        t1 = time.time()
        dt = (t1 - t0)*1000 # time difference in miliseconds
        print(f"x={data.x.shape} edge_attr={data.edge_attr.shape} norm={norm:2.2f} loss={loss.item():.4f} dt={dt:.2f}ms")

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """
    Evaluate the model on the validation set 

    Args:
        model (ProteinLigandGNN): The model to train.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for training.

    Returns:
        float: Average loss for the val set.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.ligand_x, data.ligand_batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def generate_sequence(model, data, device, temperature=1.0, top_k=None):
    """
    Generate a protein sequence using the trained model with temperature-based sampling
    and optional top-k filtering.

    Args:
        model (ProteinLigandGNN): The trained model.
        data (Data): Input data containing protein and ligand information.
        device (torch.device): Device to use for inference.
        temperature (float): Temperature for softmax. Higher values increase randomness.
        top_k (int, optional): If set, only sample from the top k most probable tokens.

    Returns:
        Tensor: Generated sequence as indices of amino acids.
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.ligand_x,
            data.ligand_batch,
        )

        # Apply temperature
        logits = logits / temperature

        if top_k is not None:
            # Get the top k logits
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=1)

            # Create a mask for the top k logits
            mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1.0)
            logits = torch.where(
                mask > 0, logits, torch.full_like(logits, float("-inf"))
            )

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Sample from the probability distribution
        sequence = torch.multinomial(probs, num_samples=1).squeeze().tolist()
        sequence_letters = list(idx_to_aa[token] for token in sequence)
        sequence_letters = list(
            IUPACData.protein_letters_3to1[resname.capitalize()]
            for resname in sequence_letters
        )

    return "".join(sequence_letters)



###############################################################################
# Main training loop                                                          #
###############################################################################

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProteinLigandGNN(
        protein_features=4,
        ligand_features=10,
        edge_features=16,
        hidden_dim=64,
        num_layers=3,
    ).to(device)

    # 3. compile model 
    model = torch.compile(model)
