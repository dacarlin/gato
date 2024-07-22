from glob import glob 
from random import shuffle, seed
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
import numpy as np 
from rich.progress import track 
from torch.utils.tensorboard import SummaryWriter


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
# Featurization                                                               #
###############################################################################


def rbf_encode(distances, num_rbf=NUM_RBF, max_distance=MAX_DISTANCE):
    """
    Encode distances using Radial Basis Functions (RBF).

    Args:
        distances (Tensor): Input distances to encode.
        num_rbf (int): Number of RBF kernels to use.
        max_distance (float): Maximum distance for RBF encoding.

    Returns:
        Tensor: RBF-encoded distances.
    """
    rbf_centers = torch.linspace(0, max_distance, num_rbf)
    rbf_widths = (max_distance / num_rbf) * torch.ones_like(rbf_centers)
    rbf_activation = torch.exp(
        -((distances.unsqueeze(-1) - rbf_centers) ** 2) / (2 * rbf_widths**2)
    )
    return rbf_activation


def phi_psi_atoms(residues):
    """
    Get the atoms necessary to compute phi and psi dihedral angles for the middle residue.

    Args:
        residues (list): A list of three consecutive Biopython residue objects.

    Returns:
        tuple: Two tuples containing atoms for phi and psi angle calculation.
               Returns (None, None) if the necessary atoms are not present.
    """
    if len(residues) != 3:
        return None, None

    phi_atoms = None
    psi_atoms = None

    if (
        "C" in residues[0]
        and "N" in residues[1]
        and "CA" in residues[1]
        and "C" in residues[1]
    ):
        phi_atoms = (
            residues[0]["C"],
            residues[1]["N"],
            residues[1]["CA"],
            residues[1]["C"],
        )

    if (
        "N" in residues[1]
        and "CA" in residues[1]
        and "C" in residues[1]
        and "N" in residues[2]
    ):
        psi_atoms = (
            residues[1]["N"],
            residues[1]["CA"],
            residues[1]["C"],
            residues[2]["N"],
        )

    return phi_atoms, psi_atoms


def calculate_dihedral(p1, p2, p3, p4):
    """
    Calculate dihedral angle between four points.

    Args:
        p1, p2, p3, p4 (Tensor): 3D coordinates of four points.

    Returns:
        float: Dihedral angle in radians.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    m1 = torch.linalg.cross(n1, b2)
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)


def get_backbone_dihedrals(residues):
    """
    Calculate backbone dihedral angles (phi, psi, omega, chi1) for a list of residues.

    Args:
        residues (list): List of Biopython residue objects.

    Returns:
        Tensor: Tensor of shape (num_residues, 4) containing dihedral angles.
    """
    dihedrals = []
    for i in range(len(residues)):
        phi, psi = phi_psi_atoms(residues[i - 1 : i + 2])

        # Calculate phi angle
        if phi is not None:
            phi_angle = calculate_dihedral(*[torch.tensor(atom.coord) for atom in phi])
        else:
            phi_angle = 0.0

        # Calculate psi angle
        if psi is not None:
            psi_angle = calculate_dihedral(*[torch.tensor(atom.coord) for atom in psi])
        else:
            psi_angle = 0.0

        # Calculate omega angle
        if i > 0:
            try:
                prev_c = residues[i - 1]["C"].coord
                curr_n = residues[i]["N"].coord
                curr_ca = residues[i]["CA"].coord
                curr_c = residues[i]["C"].coord
                omega_angle = calculate_dihedral(
                    torch.tensor(prev_c),
                    torch.tensor(curr_n),
                    torch.tensor(curr_ca),
                    torch.tensor(curr_c),
                )
            except KeyError:
                omega_angle = 0.0    
        else:
            omega_angle = 0.0

        # Calculate chi1 angle (if possible)
        if "CB" in residues[i]:
            n = residues[i]["N"].coord
            ca = residues[i]["CA"].coord
            cb = residues[i]["CB"].coord
            cg = next(
                (
                    residues[i][atom].coord
                    for atom in ["CG", "SG", "OG", "OG1"]
                    if atom in residues[i]
                ),
                None,
            )
            if cg is not None:
                chi1_angle = calculate_dihedral(
                    torch.tensor(n),
                    torch.tensor(ca),
                    torch.tensor(cb),
                    torch.tensor(cg),
                )
            else:
                chi1_angle = 0.0
        else:
            chi1_angle = 0.0

        dihedrals.append([phi_angle, psi_angle, omega_angle, chi1_angle])

    return torch.tensor(dihedrals)


def encode_atom(atom):
    """
    Encode atom features.

    Args:
        atom (Bio.PDB.Atom): Biopython atom object.

    Returns:
        Tensor: Tensor of encoded atom features.
    """
    # This is a simplified version. You may want to expand this with more detailed atom features.
    atom_types = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Other"]
    type_encoding = [1 if atom.element in atom_type else 0 for atom_type in atom_types]
    return torch.tensor(type_encoding, dtype=torch.float)


def load_protein_ligand_graph(pdb_file):
    """
    Load protein and ligand information from a PDB file and construct a graph.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        Data: A PyTorch Geometric Data object containing the graph representation.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = list(structure.get_residues())
    # remove HHOH, NA, CL, K, and other tings 
    if len(residues) > MAX_LENGTH:
        raise KeyError("too long")
    ca_atoms = [residue["CA"] for residue in residues if "CA" in residue]
    coords = torch.tensor(np.array([atom.coord for atom in ca_atoms]), dtype=torch.float)

    # Create k-nearest neighbors graph
    transform = RadiusGraph(r=MAX_DISTANCE, max_num_neighbors=MAX_NUM_NEIGHBORS, loop=True)
    data = Data(pos=coords)
    data = transform(data)

    # Compute edge features (RBF-encoded distances)
    distances = torch.norm(
        coords[data.edge_index[0]] - coords[data.edge_index[1]], dim=1
    )
    edge_attr = rbf_encode(distances)

    # Node features (backbone dihedrals)
    x = get_backbone_dihedrals(list(residue for residue in residues if "CA" in residue))

    # Target (amino acid tokens)
    amino_acids = [residue.resname for residue in residues if "CA" in residue]
    not_amino_acids = [residue.resname for residue in residues if not "CA" in residue]
    #print(not_amino_acids)
    
    #print(pdb_file, amino_acids)
    y = torch.tensor([aa_to_idx[aa] for aa in amino_acids])

    # Extract ligand and small molecule information
    ligand_atoms = [atom for atom in structure.get_atoms() if atom.parent.id[0] != " "]
    #print(ligand_atoms)

    if ligand_atoms:
        ligand_coords = torch.tensor(
            [atom.coord for atom in ligand_atoms], dtype=torch.float
        )
        ligand_features = torch.stack([encode_atom(atom) for atom in ligand_atoms])
    else:
        # Create dummy ligand data if no ligands are present
        ligand_coords = torch.zeros((1, 3), dtype=torch.float)
        ligand_features = torch.zeros((1, NUM_ATOM_FEATURES), dtype=torch.float)

    # Create a batch index for ligand atoms
    ligand_batch = torch.zeros(ligand_features.size(0), dtype=torch.long)

    assert len(x) == len(y), f"{pdb_file} {len(x)} {len(y)}"

    return Data(
        x=x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        y=y,
        ligand_x=ligand_features,
        ligand_pos=ligand_coords,
        ligand_batch=ligand_batch,
    )


###############################################################################
# Main training loop                                                          #
###############################################################################

if __name__ == "__main__":

    MAX_SAMPLES = 45_000
    pdb_files = glob("data/dompdb/*")[:MAX_SAMPLES]
    shuffle(pdb_files)
    n1 = int(0.8 * len(pdb_files))
    n2 = int(0.9 * len(pdb_files))
    train_files = pdb_files[:n1]
    val_files = pdb_files[n1:n2]
    test_files = pdb_files[n2:]

    splits = {
        "train": train_files, 
        "val": val_files, 
        "test": test_files, 
    }

    torch.save(splits, "splits.pt")

    print(f'number of examples in train={len(train_files)} val={len(val_files)} test={len(test_files)}')

    #Load training examples 
    train_set = []
    for pdb_file in track(train_files, description=f"Loading training samples (n={len(train_files)})"):
       try:
           train_set.append(load_protein_ligand_graph(pdb_file))
       except KeyError:
           pass 
    #train_set = torch.load("train_set.pt") 
    #train_loader = DataLoader(train_set, batch_size=256, shuffle=False)    
    torch.save(train_set, "train_set.pt") 

    # Evaluate on validaton set 
    val_set = []
    for pdb_file in track(val_files, description=f"Loading validation samples (n={len(val_files)})"):
        try:
            val_set.append(load_protein_ligand_graph(pdb_file))
        except KeyError:
            pass 
    torch.save(val_set, "val_set.pt") 


    # Evaluate on validaton set 
    test_set = []
    for pdb_file in track(test_files, description=f"Loading validation samples (n={len(test_files)})"):
        try:
            test_set.append(load_protein_ligand_graph(pdb_file))
        except KeyError:
            pass 
    torch.save(test_set, "test_set.pt") 
