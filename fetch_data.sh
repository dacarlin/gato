echo "Creating data folder and downloading CATH dataset."
echo  
echo "This script creates the directory tree data/dompdb"
echo "where the folder dompdb is filled with PDB files"
echo "without a file extension" 

mkdir -p data 
pushd data 

# download and create the `dompdb` folder, full of PDB files without an extension 
curl -O ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
tar -xvzf cath-nonredundant-S40.pdb.tgz
