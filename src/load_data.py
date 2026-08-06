from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, PredefinedSplit

#####################################################################################################################################################################
def create_directories(base_dir=None):
    '''
    Create directories for the project.
    input: base_dir (str or Path):  Base directory for the project. If None, uses the current file's directory.
    output: dict:                    Dictionary containing paths to the created directories.
    '''
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent                                                              #base directory is where load_data.py is

    #define and create directories for raw data, processed data, and results
    raw_data_dir = base_dir / "data" / "raw"
    processed_data_dir = base_dir / "data" / "processed"
    results_dir = base_dir / "results"

    for directory in [raw_data_dir, processed_data_dir, results_dir]:                                           #make folders if they don't exist
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "raw_data_dir": raw_data_dir,
        "processed_data_dir": processed_data_dir,
        "results_dir": results_dir,
    }

#####################################################################################################################################################################
def load_dataset(base_dir=None):
    '''
    Load the dataset.
    input:  base_dir (str or Path):     Base directory for the project. If None, uses the current file's directory.
    output: df_data (pd.DataFrame):     DataFrame containing the dataset.
    '''
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    #import processed data
    processed_data_dir = base_dir / "data" / "processed"                                                                   
    if processed_data_dir.exists():
        df_data = pd.read_csv(processed_data_dir / "clean_dataset.csv")
    else:                                                                                                                   #else get from GitHub
        csv_url = "https://raw.githubusercontent.com/SAJHillman/AIMS26/25478252292fe3bde0e4fb06977ea21c7e05545a/dataset.csv"    
        df_data = pd.read_csv(csv_url)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        df_data.to_csv(processed_data_dir / "clean_dataset.csv", index=False)

    return df_data

#####################################################################################################################################################################
def prepare_data(df_data, structure_splitting=True):
    '''
    Prepare the dataset for training by creating a list of MoleculeDatapoint objects and defining the data splitter.
    input:  df_data (pd.DataFrame):               DataFrame containing SMILES and targets
            structure_splitting (bool):           Whether to use structure-based splitting or random splitting.
    output: all_data (list):                      List of MoleculeDatapoint objects.
            splitter (PredefinedSplit or KFold):  Data splitter object for cross-validation.
            smiles (np.ndarray):                  Array of SMILES strings.
            targets (np.ndarray):                 Array of target values.
            n_splits (int):                       Number of splits used in cross-validation.    
    '''
    #extract SMILES and targets from the imported data
    from chemprop import data
    smiles = df_data.loc[:, "poly_SMI"].values
    targets = df_data.loc[:, "EA"].values

    #Use the SMILES to generate mol objects, then pair the mol objects with the targets y to make "MoleculeDatapoints"
    all_data = [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in zip(smiles, targets)]

    #Define data splitter type based on structure_splitting boolean
    if structure_splitting:                                                                             #if structure splitting, split by monomer ID            
        mA_idx = np.array([int(s.split("_")[0]) for s in df_data.loc[:, "poly_ID"]])
        splitter = PredefinedSplit(mA_idx)
        n_splits = splitter.get_n_splits()
    else:                                                                                               #else do a random k-fold split
        n_splits = 9
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=31)

    return all_data, splitter, smiles, targets, n_splits

#####################################################################################################################################################################
def build_dataloader(all_data, train_idx, valid_idx, featurizer, batch_size):
    '''
    Build the dataloaders for training and validation.
    input:  all_data (list):                        List of MoleculeDatapoint objects.
    output: train_loader (DataLoader):              Training dataloader.
            val_loader (DataLoader):                Validation dataloader.
            scaler (StandardScaler):                Scaler for normalizing targets.
    '''
    #split the data into training and validation sets *based on given the indices* i.e. random or structure-based splitting
    from chemprop import data
    train_data, val_data, _ = data.split_data_by_indices(
        data=all_data,
        train_indices=[train_idx],                                                                  #use the 2 lists of indices to split the data into training and validation sets
        val_indices=[valid_idx],
    )
    #featurise the data
    train_dset = data.MoleculeDataset(train_data[0], featurizer)                                    #MoleculeDataset is a Chemprop function that featurises the inputs. Use train_data[0] because there is some nesting
    val_dset = data.MoleculeDataset(val_data[0], featurizer)

    scaler = train_dset.normalize_targets()                                                         #normalise the targets using StandardScaler (subtract mean, scale to unit variance)
    val_dset.normalize_targets(scaler)                                              

    #build dataloaders
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=0, shuffle=False)

    return train_loader, val_loader, scaler
