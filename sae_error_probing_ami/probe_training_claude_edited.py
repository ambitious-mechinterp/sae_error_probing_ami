# %%
import torch as t
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
from transformer_lens.hook_points import HookPoint
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import warn
import os
import time  # For timing the training loops
from functools import partial
from huggingface_hub import login
from torch import Tensor
from transformer_lens.patching import get_act_patch_resid_pre,make_df_from_ranges, generic_activation_patch, layer_pos_patch_setter
import argparse
from pathlib import Path

import plotly.express as px
update_layout_set = {"xaxis_range", "yaxis_range", "yaxis2_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "xaxis_tickangle"}

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")
def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new


def imshow(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show(renderer=renderer)


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
def generate_probing_features(tokenized, model, sae, batch_size=8, device='cuda', offset=1, layer=19):
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
        layer: Layer to extract activations from
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    all_feats_input = []
    all_feats_recons = []
    all_feats_diff = []
    n = input_ids.size(0)
    
    sae_input_hook = f'blocks.{layer}.hook_resid_post.hook_sae_input'
    sae_recons_hook = f'blocks.{layer}.hook_resid_post.hook_sae_recons'
    
    for i in tqdm(range(0, n, batch_size), desc="Generating features"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in [
                sae_input_hook,
                sae_recons_hook
            ]
        )[1]
        act_input = extract_last_token_acts(batch_out[sae_input_hook], batch_mask, offset)
        act_recons = extract_last_token_acts(batch_out[sae_recons_hook], batch_mask, offset)
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
def train_probe_model(features, labels, dim, lr=1e-2, epochs=5, batch_size=8, device='cuda'):
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

def evaluate_probe_full(probe, features, labels, device='cuda'):
    """
    Evaluates the probe on the given features and labels.
    Returns the loss and accuracy.
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
    return loss.item(), accuracy

###############################################################################
# Simple Test Case for Last Token Extraction
###############################################################################
def test_last_token_extraction():
    """
    This test creates a dummy tokenized batch with padded input and a dummy
    activation tensor. It then checks that only the activations corresponding to
    the last valid token are returned.
    """
    # Create dummy tokenized batch with varying sequence lengths
    dummy_input = {
        "input_ids": t.tensor([
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 0, 0, 0],
            [8, 9, 10, 11, 12, 0]
        ]),
        "attention_mask": t.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0]
        ])
    }
    # Create a dummy activations tensor with shape (batch, seq_len, dim)
    dim = 4
    dummy_acts = t.arange(3 * 6 * dim, dtype=t.float).reshape(3, 6, dim)
    # The last valid indices for each sample are: sample0 -> index 3, sample1 -> index 2, sample2 -> index 4.
    last_acts = extract_last_token_acts(dummy_acts, dummy_input["attention_mask"])
    expected_0 = dummy_acts[0, 3, :]
    expected_1 = dummy_acts[1, 2, :]
    expected_2 = dummy_acts[2, 4, :]
    assert t.allclose(last_acts[0], expected_0), "Test failed for sample 0"
    assert t.allclose(last_acts[1], expected_1), "Test failed for sample 1"
    assert t.allclose(last_acts[2], expected_2), "Test failed for sample 2"
    print("Test last_token_extraction passed.")


# Get active latents and their activations

def record_active_latents(tokenized, model, sae, batch_size=8, device='cuda', offset=1):
    """
    Runs the model in batches and records which latents are active (nonzero) at the specified token position,
    along with their activation values.
    
    Args:
        tokenized: Tokenized input
        model: The model to run
        sae: The sparse autoencoder
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
    - List of tuples (nonzero_indices, activation_values) for each input sample
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    results = []
    n = input_ids.size(0)
    
    for i in tqdm(range(0, n, batch_size), desc="Recording active latents"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        
        # Get activations from the model
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in ['blocks.19.hook_resid_post.hook_sae_acts_post']
        )[1]
        
        # Extract specified token activations
        acts = extract_last_token_acts(
            batch_out['blocks.19.hook_resid_post.hook_sae_acts_post'], 
            batch_mask,
            offset
        )
        
        # Process each sample in the batch
        for sample_acts in acts:
            # Find nonzero indices and their values
            nonzero_mask = sample_acts != 0
            nonzero_indices = nonzero_mask.nonzero().squeeze(-1)
            nonzero_values = sample_acts[nonzero_mask]
            
            # Convert to CPU and regular Python types for storage
            results.append((
                nonzero_indices.cpu().tolist(),
                nonzero_values.cpu().tolist()
            ))
    
    return results


# %%

def run_probing_pipeline(df, tokenized_all, model, sae, device, 
                        label_columns, features_map,
                        n_seeds=50, save_probes_count=20, 
                        models_dir=None, results_dir=None,
                        layer=19, result_suffix=""):
    """
    Runs the complete probing pipeline: training probes, evaluating them,
    and computing similarities between probe weight vectors.
    
    Args:
        df: DataFrame with the dataset
        tokenized_all: Tokenized input data
        model: The model to use
        sae: The sparse autoencoder
        device: Device to use for computation
        label_columns: List of columns to use as labels
        features_map: Dictionary mapping feature types to feature tensors
        n_seeds: Number of seeds to train for
        save_probes_count: Number of seeds for which to save probes (0 for none)
        models_dir: Directory to save trained models
        results_dir: Directory to save results
        layer: Layer to train on
        result_suffix: Suffix for result filenames
    """
    # Use provided directories or defaults
    if models_dir is None:
        models_dir = Path("models")
    if results_dir is None:
        results_dir = Path("results")
    
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
    
    for seed in range(n_seeds):
        print(f"\nStarting training loop with seed {seed}...")
        start_time = time.time()
        
        # Compute a train/test split for the entire dataset using this seed
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Loop over each label
        for label_col in label_columns:
            # Prepare the labels (same for all feature types)
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
                probe, _ = train_probe_model(
                    train_feats, train_labels, dim=train_feats.size(1),
                    epochs=2, batch_size=8, device=device, lr=0.005
                )
                train_loss, train_acc = evaluate_probe_full(probe, train_feats, train_labels, device=device)
                test_loss, test_acc = evaluate_probe_full(probe, test_feats, test_labels, device=device)
                weight_norm = probe.net.weight.norm().item()
                
                probes_for_label[feature_type] = probe
                temp_results[feature_type] = {
                    "Seed": seed,
                    "Feature Type": feature_type,
                    "Label": label_col,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Weight Norm": weight_norm
                }
                
                # Save the probe if this seed is among the first N and N > 0
                if seed < save_probes_count:
                    safe_label = label_col.replace(' ', '_')
                    model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                    t.save(probe.state_dict(), models_dir / model_filename)
            
            # Compute cosine similarities between the weight vectors only if we're saving probes
            if save_probes_count > 0:
                # Extract the weight vectors (flattening them)
                w_input = probes_for_label["sae_input"].net.weight.view(-1)
                w_recons = probes_for_label["sae_recons"].net.weight.view(-1)
                w_diff = probes_for_label["sae_diff"].net.weight.view(-1)
                
                cos_sim_input_recons = F.cosine_similarity(w_input, w_recons, dim=0).item()
                cos_sim_input_diff = F.cosine_similarity(w_input, w_diff, dim=0).item()
                cos_sim_recons_diff = F.cosine_similarity(w_recons, w_diff, dim=0).item()
                
                # Add the cosine similarity metrics to each probe's result
                for feature_type in features_map.keys():
                    temp_results[feature_type]["Cosine Sim Input-Recons"] = cos_sim_input_recons
                    temp_results[feature_type]["Cosine Sim Input-Diff"] = cos_sim_input_diff
                    temp_results[feature_type]["Cosine Sim Recons-Diff"] = cos_sim_recons_diff
            
            # Append the result to the main results list
            for feature_type in features_map.keys():
                results.append(temp_results[feature_type])
            
            t.cuda.empty_cache()
        
        loop_duration = time.time() - start_time
        print(f"Training loop with seed {seed} completed in {loop_duration:.2f} seconds.")
    
    # Create a results table and print it
    results_df = pd.DataFrame(results)
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(results_csv, index=False)
    
    # Compute average cosine similarities across saved probes only if we're saving probes
    similarities_df = None
    if save_probes_count > 0:
        similarity_results = []
        # For each combination of feature type and label, load the saved probes from the first N seeds
        for label_col in label_columns:
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
                    "Feature Type": feature_type,
                    "Label": label_col,
                    "Average Cosine Similarity": avg_sim,
                    "Cosine Similarity Std": std_sim
                })
        
        similarities_df = pd.DataFrame(similarity_results)
        print("\nProbe Similarities across saved probes:")
        print(similarities_df.to_string(index=False))
        similarities_df.to_csv(similarities_csv, index=False)
    
    return results_df, similarities_df


#Setup
# %%
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train probing models on SAE activations.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gemma-2-2b", 
                        help="Model to use (e.g., gemma-2-2b)")
    parser.add_argument("--layer", type=int, default=19, 
                        help="Layer to extract activations from")
    parser.add_argument("--sae_width", type=int, default=16, 
                        help="SAE width in thousands (e.g., 16 for 16k)")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res-canonical",
                        help="SAE model release name")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="all_cities.csv",
                        help="Dataset filename in data/raw directory")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Column name for labels in the dataset")
    
    # Training arguments
    parser.add_argument("--n_seeds", type=int, default=100,
                        help="Number of random seeds to use for training")
    parser.add_argument("--save_probes_count", type=int, default=20,
                        help="Number of probes to save (0 for none)")
    
    # Experiment control
    parser.add_argument("--run_patching", action="store_true",
                        help="Whether to run the patching experiment")
    parser.add_argument("--result_suffix", type=str, default="truth",
                        help="Suffix for result files (e.g., 'truth', 'hl_frontp')")
    
    # Compute resources
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and inference")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent  # Directory of this script
    project_root = script_dir.parent    # Project root directory
    
    # Create model_sae_id for folder naming
    model_name = args.model.replace("-", "_")
    sae_id = f"{model_name}_layer_{args.layer}_width_{args.sae_width}k_canonical"
    
    # Create directories
    data_dir = project_root / "data" / "raw"
    results_dir = project_root / "results" / sae_id
    models_dir = project_root / "models" / sae_id
    
    # Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up the probing pipeline for {sae_id}")
    device = t.device(args.device)
    test_last_token_extraction()
    login()
    
    # Read datasets from data/raw
    print(f"Loading dataset {args.dataset} from {data_dir}...")
    df = pd.read_csv(data_dir / args.dataset)
    
    # Load SAE and the model
    print('Loading SAE...')
    sae_path = f"layer_{args.layer}/width_{args.sae_width}k/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=sae_path,
        device="cpu"
    )
    sae = sae.to(device)
    
    print(f'Loading Model {args.model}...')
    model = HookedSAETransformer.from_pretrained(args.model, device='cpu')
    model = model.to(device)
    
    # Tokenize the entire dataset at once
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model.tokenizer)
    label_columns = [args.label_column]
    
    # Generate features for the entire dataset
    print(f"Generating features for entire dataset from layer {args.layer}...")
    feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
        tokenized_all, model, sae, batch_size=args.batch_size, device=device, offset=1, layer=args.layer
    )
    
    # Map each feature type to its corresponding feature tensor
    features_map = {
        "sae_input": feats_all_input,
        "sae_recons": feats_all_recons,
        "sae_diff": feats_all_diff
    }
    
    # Run the probing pipeline
    print(f"Running probing pipeline with {args.n_seeds} seeds, saving {args.save_probes_count} probes...")
    results_df, similarities_df = run_probing_pipeline(
        df=df,
        tokenized_all=tokenized_all,
        model=model,
        sae=sae,
        device=device,
        label_columns=label_columns,
        features_map=features_map,
        n_seeds=args.n_seeds,
        save_probes_count=args.save_probes_count,
        models_dir=models_dir,
        results_dir=results_dir,
        layer=args.layer,
        result_suffix=args.result_suffix
    )
    
    # Clean up to free memory
    del feats_all_input, feats_all_recons, feats_all_diff, features_map
    
    # Optionally run the patching experiment if requested
    if args.run_patching:
        print("Running patching experiment to localize information...")
        clean_input = """The city of Oakland is not in the United States. This statement is: False
        The city of Canberra is in Australia. This statement is: True
        The city of Chicago is in the United States. This statement is:"""

        corrupted_input = """The city of Oakland is not in the United States. This statement is: False
        The city of Canberra is in Australia. This statement is: True
        The city of London is in the United States. This statement is:"""

        clean_tokens = model.tokenizer.encode(clean_input, return_tensors='pt').to(device)
        corrupted_tokens = model.tokenizer.encode(corrupted_input, return_tensors='pt').to(device)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        
        true_token_id = model.tokenizer.encode(" True")[1]  # Note the space before "True"
        false_token_id = model.tokenizer.encode(" False")[1]  # Note the space before "False"
        
        def patching_metric(logits):
            return logits[0, -1, true_token_id] - logits[0, -1, false_token_id]
        
        patch_results = get_act_patch_resid_pre(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=patching_metric,
        )
        
        patch_file = results_dir / f"patch_results_{args.result_suffix}.pt"
        t.save(patch_results, patch_file)
        
        # Try to visualize the results if in an interactive environment
        import sys
        is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        
        if is_interactive:
            imshow(
                patch_results,
                labels={"x": "Token Position", "y": "Layer"},
                title=f"Patching results ({args.result_suffix})",
                width=1000
            )
        else:
            print('No patching visualization in non-interactive mode')
        
        # Clean up
        t.cuda.empty_cache()
        import gc
        gc.collect()
    
    print(f"Probing completed. Results saved to {results_dir}")




# %%

def generate_steering_results(
    tokenized_text: dict,
    model: HookedSAETransformer,
    scaling_range: list,
    ref_token_1: int,
    ref_token_2: int,
    saved_probe_dir: Path,
    random_seed: int = 42,
    batch_size: int = 8,
    device: str = 'cuda',
    offset: int = 1,
    output_file: Path = None,
    label_name: str = "label",
    n_probes: int = 20,
    layer: int = 19
):
    """
    Generate steering results by applying probe weights to the model's residual stream.
    
    Args:
        saved_probe_dir: Directory containing saved probes
        tokenized_text: Tokenized input data
        model: The model to run
        scaling_range: List of scaling factors to apply
        ref_token_1: First reference token ID
        ref_token_2: Second reference token ID
        random_seed: Random seed for selecting probes
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
        output_file: Path to save results
        label_name: Name of the label column in the saved probes
        n_probes: Number of probes to use
        layer: Layer to apply steering to
    
    Returns:
        DataFrame with steering results
    """
    # Set random seed
    np.random.seed(random_seed)
    t.manual_seed(random_seed)
    
    hook_name = f'blocks.{layer}.hook_resid_post'
    
    # Load probes for each feature type
    feature_types = ['sae_input', 'sae_recons', 'sae_diff']
    probes_by_type = {}
    
    for feature_type in feature_types:
        probes_by_type[feature_type] = []
        for seed in range(n_probes):  # Load all saved probes
            safe_label = label_name.replace(' ', '_')
            model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
            filepath = saved_probe_dir / model_filename
            if filepath.exists():
                # We need to know the dimension to initialize the probe
                # For now, let's assume we can infer it from the first loaded state dict
                if len(probes_by_type[feature_type]) == 0:
                    state_dict = t.load(filepath, map_location='cpu')
                    dim = state_dict['net.weight'].size(1)
                
                # Initialize a probe and load its state dict
                probe = Probe(activation_dim=dim).to('cpu')
                probe.load_state_dict(t.load(filepath, map_location='cpu'))
                probes_by_type[feature_type].append(probe)
            else:
                print(f"Warning: Probe file {filepath} does not exist.")
    
    # Prepare input data
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    
    results = []
    n = input_ids.size(0)
    
    for i in tqdm(range(0, n, batch_size), desc="Generating steering results"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        
        # Randomly select one probe for each feature type
        selected_probes = {}
        for feature_type, probes in probes_by_type.items():
            if probes:
                selected_probes[feature_type] = probes[np.random.randint(0, len(probes))]
            else:
                print(f"Warning: No probes available for {feature_type}")
                continue
        
        # Run model without steering for baseline
        with t.no_grad():
            baseline_out = model(batch_ids)
            baseline_logits = baseline_out
            
            # Get baseline log probs for the reference tokens
            baseline_log_probs_1 = F.log_softmax(baseline_logits, dim=-1)[:, :, ref_token_1]
            baseline_log_probs_2 = F.log_softmax(baseline_logits, dim=-1)[:, :, ref_token_2]
            
            # Extract at the specific token position for each sample
            indices = get_last_token_indices(batch_mask, offset)
            batch_indices = t.arange(batch_ids.size(0), device=device)
            baseline_logprob_1 = baseline_log_probs_1[batch_indices, indices].cpu().numpy()
            baseline_logprob_2 = baseline_log_probs_2[batch_indices, indices].cpu().numpy()
            baseline_logit_diff = (baseline_logits[batch_indices, indices, ref_token_1] - 
                                  baseline_logits[batch_indices, indices, ref_token_2]).cpu().numpy()
        
        # Apply steering for each probe type and scaling factor
        for feature_type, probe in selected_probes.items():
            # Extract the weight vector from the probe
            steering_vec = probe.net.weight.view(-1).to(device)
            
            for scale in scaling_range:
                # Apply steering hook
                with t.no_grad():
                    steered_out = model.run_with_hooks(
                        batch_ids,
                        fwd_hooks=[
                            (hook_name, 
                             partial(steer_at_last_pos, 
                                     steering_vec=steering_vec, 
                                     attention_mask=batch_mask,
                                     scaling_factor=scale,
                                     offset=offset))
                        ]
                    )
                    steered_logits = steered_out
                    
                    # Get steered log probs for the reference tokens
                    steered_log_probs_1 = F.log_softmax(steered_logits, dim=-1)[:, :, ref_token_1]
                    steered_log_probs_2 = F.log_softmax(steered_logits, dim=-1)[:, :, ref_token_2]
                    
                    # Extract at the specific token position for each sample
                    steered_logprob_1 = steered_log_probs_1[batch_indices, indices].cpu().numpy()
                    steered_logprob_2 = steered_log_probs_2[batch_indices, indices].cpu().numpy()
                    steered_logit_diff = (steered_logits[batch_indices, indices, ref_token_1] - 
                                         steered_logits[batch_indices, indices, ref_token_2]).cpu().numpy()
                
                # Record results for each sample in the batch
                for j in range(batch_ids.size(0)):
                    if i + j < n:  # Ensure we're not exceeding the dataset size
                        results.append({
                            'Sample_Index': i + j,
                            'Feature_Type': feature_type,
                            'Scaling_Factor': scale,
                            'Baseline_LogProb_Token1': baseline_logprob_1[j],
                            'Baseline_LogProb_Token2': baseline_logprob_2[j],
                            'Baseline_Logit_Diff': baseline_logit_diff[j],
                            'Steered_LogProb_Token1': steered_logprob_1[j],
                            'Steered_LogProb_Token2': steered_logprob_2[j],
                            'Steered_Logit_Diff': steered_logit_diff[j],
                            'Delta_LogProb_Token1': steered_logprob_1[j] - baseline_logprob_1[j],
                            'Delta_LogProb_Token2': steered_logprob_2[j] - baseline_logprob_2[j],
                            'Delta_Logit_Diff': steered_logit_diff[j] - baseline_logit_diff[j]
                        })
        
        # Clear CUDA cache to avoid memory issues
        t.cuda.empty_cache()
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    if output_file is not None:
        results_df.to_csv(output_file, index=False)
    
    return results_df


# %%

def compute_residual_probe_dot_products(
    saved_probe_dir: Path,
    tokenized_text: dict,
    model: HookedSAETransformer,
    random_seed: int = 42,
    batch_size: int = 16,
    device: str = 'cuda',
    offset: int = 1,
    output_file: Path = None,
    label_name: str = "label",
    layer: int = 19
):
    """
    Compute dot products between residual stream values and normalized probe weights.
    
    Args:
        saved_probe_dir: Directory containing saved probes
        tokenized_text: Tokenized input data
        model: The model to run
        random_seed: Random seed for selecting probes
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
        output_file: Path to save results
        label_name: Name of the label column in the saved probes
        layer: Layer to extract activations from
    
    Returns:
        DataFrame with dot product results
    """
    # Set random seed
    np.random.seed(random_seed)
    t.manual_seed(random_seed)
    
    hook_name = f'blocks.{layer}.hook_resid_post'
    
    # Load probes for each feature type
    feature_types = ['sae_input', 'sae_recons', 'sae_diff']
    probes_by_type = {}
    
    for feature_type in feature_types:
        probes_by_type[feature_type] = []
        for seed in range(20):  # Load all 20 saved probes
            safe_label = label_name.replace(' ', '_')
            model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
            filepath = saved_probe_dir / model_filename
            if filepath.exists():
                # We need to know the dimension to initialize the probe
                if len(probes_by_type[feature_type]) == 0:
                    state_dict = t.load(filepath, map_location='cpu')
                    dim = state_dict['net.weight'].size(1)
                
                # Initialize a probe and load its state dict
                probe = Probe(activation_dim=dim).to('cpu')
                probe.load_state_dict(t.load(filepath, map_location='cpu'))
                probes_by_type[feature_type].append(probe)
            else:
                print(f"Warning: Probe file {filepath} does not exist.")
    
    # Prepare input data
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    
    results = []
    n = input_ids.size(0)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing residual-probe dot products"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        
        # Get the residual stream activations by running with cache
        with t.no_grad():
            cache = model.run_with_cache(
                batch_ids, 
                names_filter=lambda x: x == hook_name
            )[1]
            residual_activations = cache[hook_name]
        
        # Get indices of last tokens for each sample in batch
        indices = get_last_token_indices(batch_mask, offset)
        batch_indices = t.arange(batch_ids.size(0), device=device)
        
        # Extract activations at the specified positions
        # Shape: (batch_size, d_model)
        target_activations = residual_activations[batch_indices, indices]
        
        # For each sample, randomly select one probe of each type and compute dot product
        for j in range(batch_ids.size(0)):
            if i + j >= n:  # Skip if we've exceeded dataset size
                continue
                
            sample_results = {'Sample_Index': i + j}
            
            # Extract activation for this sample
            activation = target_activations[j]  # Shape: (d_model,)
            
            # Compute dot product with each probe type
            for feature_type, probes in probes_by_type.items():
                if not probes:
                    continue
                
                # Randomly select a probe
                selected_probe = probes[np.random.randint(0, len(probes))]
                
                # Get the normalized weight vector
                probe_weight = selected_probe.net.weight.view(-1).to(device)
                normalized_weight = probe_weight / probe_weight.norm()
                
                # Compute dot product
                dot_product = t.dot(activation, normalized_weight).item()
                
                # Add to results
                sample_results[f'{feature_type}_dot_product'] = dot_product
                
                # For convenience, also record the corresponding optimal scaling factor
                # (negative of dot product if we want to minimize activation in this direction)
                sample_results[f'{feature_type}_suggested_scaling'] = -dot_product
            
            results.append(sample_results)
        
        # Clear CUDA cache
        t.cuda.empty_cache()
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    if output_file is not None:
        results_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\nDot Product Summary:")
    for feature_type in feature_types:
        if f'{feature_type}_dot_product' in results_df.columns:
            mean_dot = results_df[f'{feature_type}_dot_product'].mean()
            std_dot = results_df[f'{feature_type}_dot_product'].std()
            min_dot = results_df[f'{feature_type}_dot_product'].min()
            max_dot = results_df[f'{feature_type}_dot_product'].max()
            
            print(f"{feature_type}:")
            print(f"  Mean: {mean_dot:.4f}, Std: {std_dot:.4f}")
            print(f"  Range: [{min_dot:.4f}, {max_dot:.4f}]")
            print(f"  Suggested scaling factor range: [{-max_dot:.4f}, {-min_dot:.4f}]")
    
    return results_df


# %%

# Example of running with two-shot prompting:
# python probe_training_main.py --model gemma-2-2b --layer 19 --sae_width 16 --result_suffix twoshot --n_seeds 100 --save_probes_count 25

# Example of doing steering experiments:
# python probe_training_main.py --model gemma-2-2b --layer 19 --sae_width 16 --result_suffix steering --n_seeds 5 --save_probes_count 5
# And then run a separate script that uses the saved probes for steering.

# Example of running probing on a custom dataset:
# python probe_training_main.py --model gemma-2-2b --layer 19 --sae_width 16 --dataset 114_nyc_borough_Manhattan.csv --label_column target --result_suffix manhattan
