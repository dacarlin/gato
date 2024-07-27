from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from random import seed
import torch
from torch_geometric.loader import DataLoader
from model import ProteinLigandGNN, train, evaluate, generate_sequence
from data import load_protein_ligand_graph
import argparse 


# Set seed for reproducibility 
torch.manual_seed(42)
seed(42)


# Constants
NUM_AMINO_ACIDS = 20
MAX_NUM_NEIGHBORS = 16
NUM_RBF = 16
MAX_DISTANCE = 32.0
NUM_DIHEDRAL_FEATURES = 4  # phi, psi, omega, and chi1
NUM_ATOM_FEATURES = 10  # Atom type, hybridization, aromaticity, etc.
MAX_LENGTH = 512 
HIDDEN_DIM = 128 
NUM_LAYERS = 6
BATCH_SIZE = 32 

###############################################################################
# Main training loop                                                          #
###############################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    
    # 1. switch to CUDA 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. tensor32 mm
    torch.set_float32_matmul_precision('high')

    model = ProteinLigandGNN(
        protein_features=NUM_DIHEDRAL_FEATURES,
        ligand_features=NUM_ATOM_FEATURES,
        edge_features=NUM_RBF,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"Model with {n_params:,} params")

    # 3. compile model 
    #model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/experiment_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    train_set = torch.load("train_set.pt") 
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

    # Run training 
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, device)
        writer.add_scalar("epoch/loss/train", loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # Generate sequence for a single test protein 
    #test_pdb = test_files[0]
    test_pdb = "data/dompdb/1a00B00"
    test_data = load_protein_ligand_graph(test_pdb).to(device)
    generated_sequence = generate_sequence(
        model, test_data, device, temperature=1.0, top_k=None
    )
    print("Generated sequence:", generated_sequence)

    # Evaluate on validaton set 
    val_set = torch.load("val_set.pt") 
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    loss = evaluate(model, val_loader, device)
    print(f"Val loss: {loss:4.4f}")

    # Save the model 
    torch.save(model.state_dict(), "model_state_dict.pt")
