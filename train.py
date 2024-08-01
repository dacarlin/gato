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
NUM_RBF = 16
MAX_DISTANCE = 32.0
NUM_DIHEDRAL_FEATURES = 4
NUM_ATOM_FEATURES = 10
MAX_LENGTH = 512 
HIDDEN_DIM = 64 
NUM_LAYERS = 6
BATCH_SIZE = 32 
LEARNING_RATE = 3e-4
EPOCHS = 10 
EXPR_NAME = "test"


###############################################################################
# Main training loop                                                          #
###############################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # set up cuda 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    # set up model 
    model = ProteinLigandGNN(
        protein_features=NUM_DIHEDRAL_FEATURES,
        ligand_features=NUM_ATOM_FEATURES,
        edge_features=NUM_RBF * 16,  # multiplier is 16 for feat3, 4 for feat2, 1 for feat1 
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"GAT model with {n_params:,} params")

    # set up output files 
    #model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    if EXPR_NAME:
        log_dir = f"runs/experiment_{EXPR_NAME}"
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/experiment_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    # load data 
    train_set = torch.load("train_set.pt") 
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_set = torch.load("val_set.pt") 
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Run training 
    num_epochs = EPOCHS
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        writer.add_scalar("epoch/loss/train", train_loss, epoch)
        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar("epoch/loss/val", val_loss, epoch)    
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}")

    # Save the model params 
    torch.save(model.state_dict(), "model_state_dict.pt")

    # Generate sequence for a single test protein 
    #test_pdb = test_files[0]
    test_pdb = "data/dompdb/1a00B00"
    test_data = load_protein_ligand_graph(test_pdb).to(device)
    generated_sequence = generate_sequence(
        model, test_data, device, temperature=1.0, top_k=None
    )
    print("Generated sequence:", generated_sequence)

