import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import warn
from huggingface_hub import login
import argparse
from pathlib import Path
import gc
from sklearn.metrics import roc_auc_score, roc_curve, auc
###############################################################################
# Probe Definition
###############################################################################
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

###############################################################################
# Data Helpers
###############################################################################
def train_test_split_df(df, test_size=0.2, seed=123):
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int((1 - test_size) * len(shuffled))
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]

def tokenize_data(df, tokenizer, text_column="statement"):
    texts = df[text_column].tolist()
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized


###############################################################################
# Last Token Extraction Helpers
###############################################################################
def get_last_token_indices(attention_mask, offset=1):
    """
    Given an attention mask of shape (batch, seq_len) where valid tokens are 1
    and padded tokens are 0, compute the index of the token `offset` positions from the end.
    
    Args:
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
        Tensor of indices for the specified token position
    """
    token_counts = attention_mask.sum(dim=1)
    indices = token_counts - offset
    # Make sure we don't go below 0 (if a sequence is too short)
    indices = t.clamp(indices, min=0)
    return indices

def extract_last_token_acts(act_tensor, attention_mask, offset=1):
    """
    Given a tensor of activations [batch, seq_len, dim] and the corresponding
    attention mask, select for each sample the activation at the specified token position.
    
    Args:
        act_tensor: Activation tensor of shape (batch, seq_len, dim)
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
        Tensor of activations at the specified position
    """
    indices = get_last_token_indices(attention_mask, offset)
    batch_indices = t.arange(act_tensor.size(0), device=act_tensor.device)
    activations = act_tensor[batch_indices, indices, :]
    return activations


###############################################################################
# Feature Generation
###############################################################################
def generate_probing_features(tokenized, model: HookedSAETransformer, sae: SAE, layer = 19, batch_size=8, device='cuda', offset=1):
    """
    Runs the model (with run_with_cache_with_saes) in batches on the tokenized input.
    For each batch it extracts the three features:
      - hook_sae_input, hook_sae_recons, and (sae_input - sae_recons)
    with the extraction done only at the specified token position.
    
    Args:
        tokenized: Tokenized input
        model: The model to run
        sae: The sparse autoencoder
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    all_feats_input = []
    all_feats_recons = []
    all_feats_diff = []
    n = input_ids.size(0)
    for i in tqdm(range(0, n, batch_size), desc="Generating features"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in [
                f'blocks.{layer}.hook_resid_post.hook_sae_input',
                f'blocks.{layer}.hook_resid_post.hook_sae_recons'
            ]
        )[1]
        act_input = extract_last_token_acts(batch_out[f'blocks.{layer}.hook_resid_post.hook_sae_input'], batch_mask, offset)
        act_recons = extract_last_token_acts(batch_out[f'blocks.{layer}.hook_resid_post.hook_sae_recons'], batch_mask, offset)
        act_diff = act_input - act_recons

        all_feats_input.append(act_input.detach().cpu())
        all_feats_recons.append(act_recons.detach().cpu())
        all_feats_diff.append(act_diff.detach().cpu())

    feats_input = t.cat(all_feats_input, dim=0)
    feats_recons = t.cat(all_feats_recons, dim=0)
    feats_diff = t.cat(all_feats_diff, dim=0)
    return feats_input, feats_recons, feats_diff

###############################################################################
# Probe Training and Evaluation
###############################################################################
def train_probe(features, labels, dim, lr=1e-2, epochs=5, batch_size=8, device = 'cuda'):
    """
    Trains a linear probe (a one-layer model) on the provided features to predict
    the binary labels. Returns the trained probe and a list of loss values.
    """
    probe = Probe(dim).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        probe.train()
        for batch_feats, batch_labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device).float()
            logits = probe(batch_feats)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return probe, losses

def evaluate_probe_full(probe, features, labels, device):
    """
    Evaluates the probe on the given features and labels.
    Returns the loss, accuracy, and ROC AUC score.
    """
    probe.eval()
    criterion = nn.BCEWithLogitsLoss()
    with t.no_grad():
        feats = features.to(device)
        lbls = labels.to(device).float()
        logits = probe(feats)
        loss = criterion(logits, lbls)
        preds = (logits > 0).float()
        accuracy = (preds == lbls).float().mean().item()
        
        # Calculate ROC AUC
        logits_np = logits.cpu().numpy()
        labels_np = lbls.cpu().numpy()
        try:
            roc_auc = roc_auc_score(labels_np, logits_np)
            # Get ROC curve points for potential future plotting
            fpr, tpr, _ = roc_curve(labels_np, logits_np)
            roc_curve_data = (fpr, tpr)
        except ValueError:
            # This can happen if there's only one class in the labels
            roc_auc = float('nan')
            roc_curve_data = None
            
    return loss.item(), accuracy, roc_auc, roc_curve_data



def run_probing_pipeline(df: pd.DataFrame, device: t.device, 
                        label_col: str, features_map: dict[str, t.Tensor],
                        models_dir: Path, results_dir: Path,
                        epochs = 2, lr = 0.005, batch_size = 16,
                        n_seeds=50, save_probes_count=20, 
                        result_suffix=""):
    """
    Runs the complete probing pipeline: training probes, evaluating them,
    and computing similarities between probe weight vectors.
    
    Args:
        df: DataFrame with the dataset
        device: Device to use for computation
        label_col: Column to use as labels
        features_map: Dictionary mapping feature types to feature tensors
        n_seeds: Number of seeds to train for
        save_probes_count: Number of seeds for which to save probes (0 for none)
        models_dir: Directory to save trained models
        results_dir: Directory to save results
        result_suffix: Suffix for result filenames
    """
    # Use provided directories or defaults
    
    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct result filenames with suffix
    suffix = f"_{result_suffix}" if result_suffix else ""
    results_csv = results_dir / f"probe_results{suffix}.csv"
    similarities_csv = results_dir / f"probe_similarities{suffix}.csv"
    
    results = []
    n = df.shape[0]
    train_size = int(0.8 * n)  # 80% for training
    
    for seed in tqdm(range(n_seeds), desc="Training probes"):
        # Compute a train/test split for the entire dataset using this seed
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        labels_all = t.tensor(df[label_col].values)
        train_labels = labels_all[train_indices]
        test_labels = labels_all[test_indices]
        
        # Dictionary to hold the probes for the three feature types for cosine similarity computation
        probes_for_label = {}
        # Temporary storage for the per-probe results for this seed & label
        temp_results = {}
        
        # Train a probe for each feature type
        for feature_type, feats_all in features_map.items():
            train_feats = feats_all[train_indices]
            test_feats = feats_all[test_indices]
            
            # Set the torch seed to ensure probe initialization consistency
            t.manual_seed(seed)
            probe, _ = train_probe(
                train_feats, train_labels, dim=train_feats.size(1),
                epochs=epochs, batch_size=batch_size, device=device, lr=lr
            )
            train_loss, train_acc, train_roc_auc, _ = evaluate_probe_full(probe, train_feats, train_labels, device=device)
            test_loss, test_acc, test_roc_auc, _ = evaluate_probe_full(probe, test_feats, test_labels, device=device)
            weight_norm = probe.net.weight.norm().item()
            
            probes_for_label[feature_type] = probe
            temp_results[feature_type] = {
                "Seed": seed,
                "Feature_Type": feature_type,
                "Label": label_col,
                "Train_Loss": train_loss,
                "Train_Accuracy": train_acc,
                "Train_ROC_AUC": train_roc_auc,
                "Test_Loss": test_loss,
                "Test_Accuracy": test_acc,
                "Test_ROC_AUC": test_roc_auc,
                "Weight_Norm": weight_norm
            }
            
            # Save the probe if this seed is among the first N and N > 0
            if seed < save_probes_count:
                safe_label = label_col.replace(' ', '_')
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                t.save(probe.state_dict(), models_dir / model_filename)

        # Extract the weight vectors (flattening them)
        w_input = probes_for_label["sae_input"].net.weight.view(-1)
        w_recons = probes_for_label["sae_recons"].net.weight.view(-1)
        w_diff = probes_for_label["sae_diff"].net.weight.view(-1)
        
        cos_sim_input_recons = F.cosine_similarity(w_input, w_recons, dim=0).item()
        cos_sim_input_diff = F.cosine_similarity(w_input, w_diff, dim=0).item()
        cos_sim_recons_diff = F.cosine_similarity(w_recons, w_diff, dim=0).item()
        
        # Add the cosine similarity metrics to each probe's result
        for feature_type in features_map.keys():
            #This is just repeated. 
            temp_results[feature_type]["Cosine_Sim_Input_Recons"] = cos_sim_input_recons
            temp_results[feature_type]["Cosine_Sim_Input_Diff"] = cos_sim_input_diff
            temp_results[feature_type]["Cosine_Sim_Recons_Diff"] = cos_sim_recons_diff
            results.append(temp_results[feature_type])
        
        t.cuda.empty_cache()
    
    # Create a results table and print it
    results_df = pd.DataFrame(results)
    print(f"{result_suffix} finished training {n_seeds} probes.")
    print('Average Test Loss')
    print(results_df.groupby('Feature_Type')['Test_Loss'].mean())
    print('Average Test Accuracy')
    print(results_df.groupby('Feature_Type')['Test_Accuracy'].mean())
    print('Average Test ROC AUC')
    print(results_df.groupby('Feature_Type')['Test_ROC_AUC'].mean())
    better_loss = np.mean(
        results_df[results_df["Feature_Type"] == "sae_diff"]["Test_Loss"].to_numpy() < results_df[results_df["Feature_Type"] == "sae_input"]["Test_Loss"].to_numpy())
    print(f"""Proportion of sae error nodes with better loss: {better_loss:3f}""")
    better_auc = np.mean(
        results_df[results_df["Feature_Type"] == "sae_diff"]["Test_ROC_AUC"].to_numpy() > results_df[results_df["Feature_Type"] == "sae_input"]["Test_ROC_AUC"].to_numpy())
    print(f"""Proportion of sae error nodes with better ROC AUC: {better_auc:3f}""")

    results_df.to_csv(results_csv, index=False)
    
    # Compute average cosine similarities across saved probes only if we're saving probes
    if save_probes_count > 0:
        print("Computing average cosine similarities across saved probes...")
        similarity_results = []
        safe_label = label_col.replace(' ', '_')
        for feature_type in features_map.keys():
            weight_vectors = []
            for seed in range(save_probes_count):
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                filepath = models_dir / model_filename
                if filepath.exists():
                    # Initialize a probe and load its state dict
                    dummy_probe = Probe(activation_dim=features_map[feature_type].size(1)).to('cpu')
                    state_dict = t.load(filepath, map_location='cpu')
                    dummy_probe.load_state_dict(state_dict)
                    weight_vectors.append(dummy_probe.net.weight.view(-1))
                else:
                    warn(f"Probe file {filepath} does not exist.")
            
            # Compute pairwise cosine similarities among these weight vectors
            sims = []
            num = len(weight_vectors)
            for i in range(num):
                for j in range(i+1, num):
                    sim = F.cosine_similarity(weight_vectors[i], weight_vectors[j], dim=0).item()
                    sims.append(sim)
            if sims:
                avg_sim = np.mean(sims)
                std_sim = np.std(sims)
            else:
                avg_sim = None
                std_sim = None
            similarity_results.append({
                "Feature_Type": feature_type,
                "Label": label_col,
                "Average_Cosine_Similarity": avg_sim,
                "Cosine_Similarity_Std": std_sim
            })
    
        similarities_df = pd.DataFrame(similarity_results)
        similarities_df.to_csv(similarities_csv, index=False)
    
    return results_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train probing models on SAE activations.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gemma-2-2b", 
                        help="Model to use (e.g., gemma-2-2b)")
    parser.add_argument("--layer", type=int, default=19, 
                        help="Layer to extract activations from")
    parser.add_argument("--sae_id", type=str, default="layer_19/width_16k/canonical", 
                        help="SAE id for SAELens")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res-canonical",
                        help="SAE model release name")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, nargs = "+", default="[all_cities.csv]",
                        help="Dataset filename in data/raw directory. Accepts multiple datasets")
    parser.add_argument("--label_column", type=str, default="target",
                        help="Column name for labels in the dataset")
    parser.add_argument("--token_position_offset", type = int, default = 1,
                        help="Offset from last token position. Default 1 is last token, 2 is second to last, etc.")
    
    # Training arguments
    parser.add_argument("--n_seeds", type=int, default=100,
                        help="Number of random seeds to use for training")
    parser.add_argument("--save_probes_count", type=int, default=0,
                        help="Number of probes to save (0 for none)")
    
    # Experiment control
    parser.add_argument("--run_patching", action="store_true",
                        help="Whether to run the patching experiment")
    parser.add_argument("--result_suffix", type=str, nargs = "+", default=["truth"],
                        help="Suffix for result files (e.g., 'truth', 'hl_frontp')")
    
    # Compute resources
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and inference")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    
    #Hyperparameters
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for training")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Set up paths
    script_dir = Path(__file__).parent  # Directory of this script
    project_root = script_dir.parent    # Project root directory

    print('Setting up pipeline')
    login()
    device = t.device(args.device)
    # Load SAE and the model
    print('Load SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device="cpu"
    )
    sae = sae.to(device)
    print('Load Model')
    model = HookedSAETransformer.from_pretrained(args.model, device='cpu')
    model = model.to(device)

    if type(args.dataset) == str:
        args.dataset = [args.dataset]

    if type(args.result_suffix) == str:
        args.result_suffix = [args.result_suffix]

    assert len(args.dataset) == len(args.result_suffix), "Dataset and result suffix must have the same length"

    for dataset, result_suffix in zip(args.dataset, args.result_suffix):
        gc.collect()
        print(f"Processing dataset: {dataset}")
        df = pd.read_csv(project_root / "data" / "raw" / dataset)
        print("Tokenizing entire dataset...")
        tokenized_all = tokenize_data(df, model.tokenizer)
        label = args.label_column
        features_possibilities = ['sae_input', 'sae_recons', 'sae_diff']
        print(f"Generating features for entire dataset with offset: {args.token_position_offset}")
        feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
            tokenized_all, model, sae, batch_size=args.batch_size, device=device, offset=args.token_position_offset
        )
        features_map = {
            "sae_input": feats_all_input,
            "sae_recons": feats_all_recons,
            "sae_diff": feats_all_diff
        }
        model_sae_id = args.model.replace("-", "_") + "_" + args.sae_id.replace("/", "_").replace("-","_")
        results_df = run_probing_pipeline(df, device, label, features_map, project_root / "models" / model_sae_id, project_root / "results" / model_sae_id, args.epochs, args.lr, args.batch_size, args.n_seeds, args.save_probes_count, result_suffix)
        t.cuda.empty_cache()
        
        
    
    

    
    
    