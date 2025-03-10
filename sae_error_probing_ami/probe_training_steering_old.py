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
def generate_probing_features(tokenized, model, sae, batch_size=8, device='cuda', offset=1):
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
                'blocks.19.hook_resid_post.hook_sae_input',
                'blocks.19.hook_resid_post.hook_sae_recons'
            ]
        )[1]
        act_input = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_input'], batch_mask, offset)
        act_recons = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_recons'], batch_mask, offset)
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
                        probe_save_dir="trained_probes", 
                        results_csv="probe_results.csv",
                        similarities_csv="probe_similarities.csv"):
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
        save_probes_count: Number of seeds for which to save probes
        probe_save_dir: Directory to save trained probes
        results_csv: Filename for the results CSV
        similarities_csv: Filename for the similarities CSV
    """
    os.makedirs(probe_save_dir, exist_ok=True)
    
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
                
                # Save the probe if this seed is among the first N
                if seed < save_probes_count:
                    safe_label = label_col.replace(' ', '_')
                    model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                    t.save(probe.state_dict(), os.path.join(probe_save_dir, model_filename))
            
            # Compute cosine similarities between the weight vectors for the three probes
            # Extract the weight vectors (flattening them)
            w_input = probes_for_label["sae_input"].net.weight.view(-1)
            w_recons = probes_for_label["sae_recons"].net.weight.view(-1)
            w_diff = probes_for_label["sae_diff"].net.weight.view(-1)
            
            cos_sim_input_recons = F.cosine_similarity(w_input, w_recons, dim=0).item()
            cos_sim_input_diff = F.cosine_similarity(w_input, w_diff, dim=0).item()
            cos_sim_recons_diff = F.cosine_similarity(w_recons, w_diff, dim=0).item()
            
            # Add the cosine similarity metrics to each probe's result
            for feature_type in features_map.keys():
                #This 
                temp_results[feature_type]["Cosine Sim Input-Recons"] = cos_sim_input_recons
                temp_results[feature_type]["Cosine Sim Input-Diff"] = cos_sim_input_diff
                temp_results[feature_type]["Cosine Sim Recons-Diff"] = cos_sim_recons_diff
                
                # Append the result to the main results list
                results.append(temp_results[feature_type])
            
            t.cuda.empty_cache()
        
        loop_duration = time.time() - start_time
        print(f"Training loop with seed {seed} completed in {loop_duration:.2f} seconds.")
    
    # Create a results table and print it
    results_df = pd.DataFrame(results)
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(results_csv, index=False)
    
    # Compute average cosine similarities across saved probes
    similarity_results = []
    # For each combination of feature type and label, load the saved probes from the first N seeds
    for label_col in label_columns:
        safe_label = label_col.replace(' ', '_')
        for feature_type in features_map.keys():
            weight_vectors = []
            for seed in range(save_probes_count):
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                filepath = os.path.join(probe_save_dir, model_filename)
                if os.path.exists(filepath):
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
if __name__ == "__main__":
    print("Setting up the probing pipeline")
    device = t.device('cuda:0')
    test_last_token_extraction()
    login()
    
    # Read datasets and combine them.
    df = pd.read_csv("all_cities.csv")
    
    # Load SAE and the model.
    print('Load SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_19/width_16k/canonical",
        device="cpu"
    )
    sae = sae.to(device)
    print('Load Model')
    model2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device='cpu')
    model2b = model2b.to(device)
    
    # Tokenize the entire dataset at once.
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model2b.tokenizer)
    label_columns = ['label']
    
    features_possibilities = ['sae_input', 'sae_recons', 'sae_diff']

        # Directory to save trained probes.
    probe_save_dir = "trained_probes_truth"
    os.makedirs(probe_save_dir, exist_ok=True)
    



# %%
if __name__ == "__main__":
    # Run the original probing pipeline on last token (offset=1)
    print("Running probing pipeline on LAST tokens...")
    
    # Generate features for the entire dataset
    print("Generating features for entire dataset (last token)...")
    feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device=device, offset=1
    )
    
    # Map each feature type to its corresponding feature tensor
    features_map_last = {
        "sae_input": feats_all_input,
        "sae_recons": feats_all_recons,
        "sae_diff": feats_all_diff
    }
    
    # Run the probing pipeline
    results_df_last, similarities_df_last = run_probing_pipeline(
        df=df,
        tokenized_all=tokenized_all,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=label_columns,
        features_map=features_map_last,
        n_seeds=100,
        save_probes_count=20,
        probe_save_dir="trained_probes_truth",
        results_csv="probe_results_truth.csv",
        similarities_csv="probe_similarities_truth.csv"
    )
    del feats_all_input, feats_all_recons, feats_all_diff, features_map_last

# %%
if __name__ == "__main__":
    # Run the probing pipeline on second-to-last tokens (offset=2)
    print("Running probing pipeline on SECOND-TO-LAST tokens...")
    
    # Generate features for the entire dataset using second-to-last tokens
    print("Generating features for entire dataset (second-to-last token)...")
    feats_all_input_2nd, feats_all_recons_2nd, feats_all_diff_2nd = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device=device, offset=2
    )
    
    # Map each feature type to its corresponding feature tensor
    features_map_2nd = {
        "sae_input": feats_all_input_2nd,
        "sae_recons": feats_all_recons_2nd,
        "sae_diff": feats_all_diff_2nd
    }
    
    # Run the probing pipeline
    results_df_2nd, similarities_df_2nd = run_probing_pipeline(
        df=df,
        tokenized_all=tokenized_all,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=label_columns,
        features_map=features_map_2nd,
        n_seeds=100,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_second_last",
        results_csv="probe_results_truth_second_last.csv",
        similarities_csv="probe_similarities_truth_second_last.csv"
    )
    del feats_all_input_2nd, feats_all_recons_2nd, feats_all_diff_2nd, features_map_2nd
# %%
#Run probe on twitter happiness

if __name__ == "__main__":
    print("Running probing pipeline on TWITTER HAPPINESS...")
    tw_happiness = pd.read_csv('149_twt_emotion_happiness.csv')
    tw_happiness_tokenized = tokenize_data(tw_happiness, model2b.tokenizer, 'prompt')

    feats_tw_input, feats_tw_recons, feats_tw_diff = generate_probing_features(
        tw_happiness_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_tw = {
        "sae_input": feats_tw_input,
        "sae_recons": feats_tw_recons,
        "sae_diff": feats_tw_diff
    }
    
    # Run the probing pipeline
    results_df_tw, similarities_df_tw = run_probing_pipeline(
        df=tw_happiness,
        tokenized_all=tw_happiness_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_tw,
        n_seeds=100,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_tw",
        results_csv="probe_results_tw_happiness.csv",
        similarities_csv="probe_similarities_tw_happiness.csv"
    )
    results_df_tw.groupby(['Feature Type', 'Label'])['Test Accuracy'].mean().reset_index()
    del feats_tw_input, feats_tw_recons, feats_tw_diff, features_map_tw
    t.cuda.empty_cache()
    import gc
    gc.collect()
    
# %%
#Run probe on headline front page

if __name__ == "__main__":
    print("Running probing pipeline on HEADLINE FRONT PAGE...")
    hl_frontp = pd.read_csv('headline_frontpage_sample.csv')
    hl_frontp_tokenized = tokenize_data(hl_frontp, model2b.tokenizer, 'prompt')

    feats_hl_input, feats_hl_recons, feats_hl_diff = generate_probing_features(
        hl_frontp_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_hl = {
        "sae_input": feats_hl_input,
        "sae_recons": feats_hl_recons,
        "sae_diff": feats_hl_diff
    }
    
    # Run the probing pipeline
    results_df_hl, similarities_df_hl = run_probing_pipeline(
        df=hl_frontp,
        tokenized_all=hl_frontp_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_hl,
        n_seeds=100,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_hl",
        results_csv="probe_results_hl_frontp.csv",
        similarities_csv="probe_similarities_hl_frontp.csv"
    )
    del feats_hl_input, feats_hl_recons, feats_hl_diff, features_map_hl

# %%
#Run probe on if the borough is manhattan

if __name__ == "__main__":
    print("Running probing pipeline on MANHATTAN...")
    manhattan = pd.read_csv('114_nyc_borough_Manhattan.csv')
    manhattan_tokenized = tokenize_data(manhattan, model2b.tokenizer, 'prompt')

    feats_man_input, feats_man_recons, feats_man_diff = generate_probing_features(
        manhattan_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_man = {
        "sae_input": feats_man_input,
        "sae_recons": feats_man_recons,
        "sae_diff": feats_man_diff
    }
    
    # Run the probing pipeline
    results_df_man, similarities_df_man = run_probing_pipeline(
        df=manhattan,
        tokenized_all=manhattan_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_man,
        n_seeds=100,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_man",
        results_csv="probe_results_man_borough.csv",
        similarities_csv="probe_similarities_man_borough.csv"
    )
    del feats_man_input, feats_man_recons, feats_man_diff, features_map_man
# %%
# run probe on athlete sport basketball 155_athlete_sport_basketball.csv

if __name__ == "__main__":
    print("Running probing pipeline on ATHLETE SPORT BASKETBALL...")
    athlete_sport = pd.read_csv('155_athlete_sport_basketball.csv')
    athlete_sport_tokenized = tokenize_data(athlete_sport, model2b.tokenizer, 'prompt')
    
    feats_ath_input, feats_ath_recons, feats_ath_diff = generate_probing_features(
        athlete_sport_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_ath = {
        "sae_input": feats_ath_input,
        "sae_recons": feats_ath_recons,
        "sae_diff": feats_ath_diff
    }
    
    # Run the probing pipeline
    results_df_ath, similarities_df_ath = run_probing_pipeline(
        df=athlete_sport,
        tokenized_all=athlete_sport_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_ath,
        n_seeds=100,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_ath",
        results_csv="probe_results_ath_sport.csv",
        similarities_csv="probe_similarities_ath_sport.csv"
    )
    results_df_ath.groupby(['Feature Type', 'Label'])['Test Loss'].mean().reset_index()
    del feats_ath_input, feats_ath_recons, feats_ath_diff, features_map_ath
    t.cuda.empty_cache()
    




# %%

#Probe steering
if __name__ == "__main__":
    print("Getting steering functions")
def steer_at_last_pos(
    input_resid: t.Tensor,
    hook: HookPoint,
    steering_vec: t.Tensor,
    attention_mask: t.Tensor,
    scaling_factor: float = 1.0,
    offset: int = 1
) -> t.Tensor:
    """
    Apply steering to the residual stream at the specified token position.
    
    Args:
        input_resid: Residual stream tensor of shape (batch, seq_len, d_model)
        hook: HookPoint
        steering_vec: Steering vector of shape (d_model)
        attention_mask: Attention mask of shape (batch, seq_len)
        scaling_factor: Scaling factor for the steering vector
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
        Modified residual stream tensor
    """
    # Create a new tensor instead of modifying in-place
    new_input_resid = input_resid.clone()
    
    # Get indices of the specified token positions
    indices = get_last_token_indices(attention_mask, offset)
    batch_indices = t.arange(input_resid.size(0), device=input_resid.device)
    
    # Normalize steering vector to unit norm
    vec_norm = steering_vec / steering_vec.norm()
    
    # Apply steering only at the specified positions
    new_input_resid[batch_indices, indices] = new_input_resid[batch_indices, indices] + vec_norm * scaling_factor
    
    return new_input_resid

def generate_steering_results(
    tokenized_text: dict,
    model: HookedSAETransformer,
    scaling_range: list,
    ref_token_1: int,
    ref_token_2: int,
    saved_probe_dir:str = "trained_probes_truth",
    random_seed: int = 42,
    batch_size: int = 8,
    device: str = 'cuda',
    offset: int = 1,
    output_csv: str = "steering_results.csv",
    label_name: str = "label",
    n_probes: int = 20
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
        output_csv: Filename to save results
        label_name: Name of the label column in the saved probes
    
    Returns:
        DataFrame with steering results
    """
    # Set random seed
    np.random.seed(random_seed)
    t.manual_seed(random_seed)
    
    # Load probes for each feature type
    feature_types = ['sae_input', 'sae_recons', 'sae_diff']
    probes_by_type = {}
    
    for feature_type in feature_types:
        probes_by_type[feature_type] = []
        for seed in range(n_probes):  # Load all 20 saved probes
            safe_label = label_name.replace(' ', '_')
            model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
            filepath = os.path.join(saved_probe_dir, model_filename)
            if os.path.exists(filepath):
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
                            ('blocks.19.hook_resid_post', 
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
    results_df.to_csv(output_csv, index=False)
    
    return results_df

def compute_residual_probe_dot_products(
    saved_probe_dir: str,
    tokenized_text: dict,
    model: HookedSAETransformer,
    random_seed: int = 42,
    batch_size: int = 16,
    device: str = 'cuda',
    offset: int = 1,
    output_csv: str = "residual_probe_dot_products.csv",
    label_name: str = "label"
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
        output_csv: Filename to save results
        label_name: Name of the label column in the saved probes
    
    Returns:
        DataFrame with dot product results
    """
    # Set random seed
    np.random.seed(random_seed)
    t.manual_seed(random_seed)
    
    # Load probes for each feature type
    feature_types = ['sae_input', 'sae_recons', 'sae_diff']
    probes_by_type = {}
    
    for feature_type in feature_types:
        probes_by_type[feature_type] = []
        for seed in range(20):  # Load all 20 saved probes
            safe_label = label_name.replace(' ', '_')
            model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
            filepath = os.path.join(saved_probe_dir, model_filename)
            if os.path.exists(filepath):
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
                names_filter=lambda x: x == 'blocks.19.hook_resid_post'
            )[1]
            residual_activations = cache['blocks.19.hook_resid_post']
        
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
    results_df.to_csv(output_csv, index=False)
    
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

#Also some patching utilty


def get_act_patch_specific_positions(
    model, corrupted_tokens, clean_cache, patching_metric, 
    activation_name="resid_pre", patch_positions=None
):
    """
    Function to get activation patching results for only specific positions.
    
    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric
        activation_name: The name of the activation to patch
        patch_positions: List of positions to patch (if None, all positions are patched)
    
    Returns:
        patched_output: The tensor of the patching metric for each patch
    """
    # Create a dataframe with all possible layer, position combinations
    all_indices = make_df_from_ranges(
        [model.cfg.n_layers, corrupted_tokens.shape[-1]], 
        ["layer", "pos"]
    )
    
    # Filter to only include specified positions
    if patch_positions is not None:
        filtered_indices = all_indices[all_indices["pos"].isin(patch_positions)]
    else:
        filtered_indices = all_indices
    
    # Use the filtered dataframe with generic_activation_patch
    return generic_activation_patch(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        patching_metric=patching_metric,
        patch_setter=layer_pos_patch_setter,
        activation_name=activation_name,
        index_df=filtered_indices
    )

# %%
#Patching to localize information
if __name__ == "__main__":
    print("Patching to localize information")
    clean_input = """The city of Oakland is not in the United States. This statement is: False
    The city of Canberra is in Australia. This statement is: True
    The city of Chicago is in the United States. This statement is:"""

    corrupted_input = """The city of Oakland is not in the United States. This statement is: False
    The city of Canberra is in Australia. This statement is: True
    The city of London is in the United States. This statement is:"""

    clean_tokens = model2b.tokenizer.encode(clean_input, return_tensors='pt').to(device)
    corrupted_tokens = model2b.tokenizer.encode(corrupted_input, return_tensors='pt').to(device)
    clean_logits, clean_cache = model2b.run_with_cache(clean_tokens)
    print(t.topk(F.softmax(clean_logits[0,-1]), k = 10))
    true_token_id = model2b.tokenizer.encode(" True")[1]  # Note the space before "True"
    false_token_id = model2b.tokenizer.encode(" False")[1]  # Note the space before "False"

    def patching_metric(logits):
        return logits[0, -1, true_token_id] - logits[0, -1, false_token_id]
    

    patch_results = get_act_patch_resid_pre(
        model= model2b,
        corrupted_tokens = corrupted_tokens,
        clean_cache = clean_cache,
        patching_metric = patching_metric,
    )
    t.save(patch_results, "patch_results.pt")
    t.cuda.empty_cache()
    import sys
    is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
    
    if is_interactive:
        imshow(
            patch_results,
            labels={"x": "Token Position", "y": "Layer"},
            title="Patching results for the truthfulness information",
            width=1000
        )
    else:
        print('No patching visualization in non-interactive mode')

    gc.collect()



#%%
#Generate probes for the two shot prompted data
if __name__ == "__main__":
    twoshot_tokenized = tokenize_data(df, model2b.tokenizer, 'twoshot_prompt')
    print("Generating probing results for the two shot prompted data")
    feats_twoshot_input, feats_twoshot_recons, feats_twoshot_diff = generate_probing_features(
        twoshot_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_twoshot = {
        "sae_input": feats_twoshot_input,
        "sae_recons": feats_twoshot_recons,
        "sae_diff": feats_twoshot_diff
    }
    
    # Run the probing pipeline
    results_df_twoshot, similarities_df_twoshot = run_probing_pipeline(
        df=df,
        tokenized_all=twoshot_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['label'],
        features_map=features_map_twoshot,
        n_seeds=100,
        save_probes_count=25,
        probe_save_dir="trained_probes_truth_twoshot",
        results_csv="probe_results_twoshot.csv",
        similarities_csv="probe_similarities_twoshot.csv"
    )
    results_df_twoshot.groupby(['Feature Type', 'Label'])['Test Loss'].mean().reset_index()
    del feats_twoshot_input, feats_twoshot_recons, feats_twoshot_diff, features_map_twoshot



# %%
#Checking the dot products
if __name__ == "__main__":
    print("Checking the dot products")
    prompted_tokenized = tokenize_data(df.sample(128), model2b.tokenizer, 'twoshot_prompt')
    # Compute dot products between residual stream and probe weights
    dot_products = compute_residual_probe_dot_products(
        saved_probe_dir="trained_probes_truth_twoshot",
        tokenized_text=prompted_tokenized,
        model=model2b,
        random_seed=42,
        batch_size=16,
        device=device,
        offset=1,  # Last token
        output_csv="residual_probe_dot_products.csv",
        label_name="label"
    )

    # View the distribution of dot products
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    feature_types = ['sae_input', 'sae_recons', 'sae_diff']

    for i, feature_type in enumerate(feature_types):
        if f'{feature_type}_dot_product' in dot_products.columns:
            axes[i].hist(dot_products[f'{feature_type}_dot_product'], bins=30)
            axes[i].set_title(f'{feature_type} Dot Products')
            axes[i].set_xlabel('Dot Product Value')
            axes[i].set_ylabel('Frequency')
            
            # Add vertical line at mean
            mean_val = dot_products[f'{feature_type}_dot_product'].mean()
            axes[i].axvline(mean_val, color='r', linestyle='--', 
                        label=f'Mean: {mean_val:.4f}')
            axes[i].legend()

    plt.tight_layout()
    if is_interactive:
        plt.show()
    else:
        # Save the figure instead of displaying it
        print("Running in non-interactive mode. Saving figure to 'dot_products_histogram.png'")
        plt.savefig('dot_products_histogram.png')
        plt.close(fig)
    t.cuda.empty_cache()




# %%
#Applying steering
if __name__ == "__main__":
    # Example usage of the steering function
    scaling_range = [-20,-10,-8,-5.0, 5.0, 8, 10, 20]
    print("Generating steering results")
    # Get token IDs for "True" and "False"
      # Note the space before "False"
    np.random.seed(32)
    df_sample = df.sample(480)
    df_sample.to_csv("df_sample.csv", index=False)
    sample_tokenized = tokenize_data(df_sample, model2b.tokenizer, 'twoshot_prompt')
    # Generate steering results
    steering_results = generate_steering_results(
        saved_probe_dir="trained_probes_truth_twoshot",
        tokenized_text=sample_tokenized,
        model=model2b,
        scaling_range=scaling_range,
        ref_token_1=true_token_id,
        ref_token_2=false_token_id,
        random_seed=32,
        batch_size=16,  # Smaller batch size for steering
        device=device,
        offset=1,  # Last token
        output_csv="steering_results_truth.csv",
        label_name="label",
        n_probes=25
    )


# %%
