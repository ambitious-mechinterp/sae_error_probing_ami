import torch as t
from torch import Tensor
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
from transformers import PreTrainedTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import warn
from huggingface_hub import login
import argparse
from typing import Any
from numpy.typing import NDArray
from pathlib import Path
import gc
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from utils import tokenize_data, extract_last_token_acts
from jaxtyping import Float


###############################################################################
# Feature Generation
###############################################################################
def generate_probing_features(tokenized: dict[str, Float[Tensor, "batch seq_len"]], model: HookedSAETransformer, 
                              sae: SAE, layer: int = 19, batch_size: int = 8, device: t.device | str = 'cuda', offset: int = 1
                              ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_model"], Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae_hidden"]]:
    """
    Gather activations for residual stream, SAE activations, SAE error, SAE reconstruction.

    Args:
        tokenized: Tokenized input data.
        model: Model to use for feature extraction.
        sae: SAE to use for feature extraction.
        layer: Layer to extract activations from.
        batch_size: Batch size for feature extraction.
        device: Device to use for feature extraction.
        offset: Offset from the last token to extract activations from.
    
    Returns:
        Tuple of tensors (input, recons, error, acts_post_nonlinearity)
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    all_feats_input = []
    all_feats_recons = []
    all_feats_diff = []
    all_feats_acts_post = []
    n = input_ids.size(0)
    for i in tqdm(range(0, n, batch_size), desc="Generating features"):
        t.cuda.empty_cache()
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in [
                f'blocks.{layer}.hook_resid_post.hook_sae_input',
                f'blocks.{layer}.hook_resid_post.hook_sae_recons',
                f'blocks.{layer}.hook_resid_post.hook_sae_acts_post'
            ]
        )[1]
        act_input = extract_last_token_acts(batch_out[f'blocks.{layer}.hook_resid_post.hook_sae_input'], batch_mask, offset)
        act_recons = extract_last_token_acts(batch_out[f'blocks.{layer}.hook_resid_post.hook_sae_recons'], batch_mask, offset)
        act_diff = act_input - act_recons
        act_acts_post = extract_last_token_acts(batch_out[f'blocks.{layer}.hook_resid_post.hook_sae_acts_post'], batch_mask, offset)
        all_feats_input.append(act_input.detach().cpu())
        all_feats_recons.append(act_recons.detach().cpu())
        all_feats_diff.append(act_diff.detach().cpu())
        all_feats_acts_post.append(act_acts_post.detach().cpu())

    feats_input = t.cat(all_feats_input, dim=0)
    feats_recons = t.cat(all_feats_recons, dim=0)
    feats_diff = t.cat(all_feats_diff, dim=0)
    feats_acts_post = t.cat(all_feats_acts_post, dim=0)
    return feats_input, feats_recons, feats_diff, feats_acts_post

###############################################################################
# Probe Training and Evaluation
###############################################################################
def train_logistic_probe(features: Float[Tensor, "batch d_model"], labels: Float[Tensor, "batch"], epochs: int = 2, 
                batch_size: int = 8, penalty: str = 'l2', C: float = 0.1) -> LogisticRegression:
    """
    Trains a logistic regression model with the specified penalty (l2 by default, l1 for SAE acts)
    on the provided features to predict the binary labels. Returns the trained model.
    """
    # Convert features and labels to numpy arrays for sklearn
    features_np = features.to(t.float32).detach().cpu().numpy()
    labels_np = labels.to(t.float32).detach().cpu().numpy()

    model = LogisticRegression(
        penalty=penalty,
        C=C,  # C=0.1 corresponds to a regularization strength of lambda=10 (1/C)
        max_iter=epochs * len(features_np) // batch_size + 100,
        random_state=None,
        solver='liblinear',
        verbose=0
    )

    model.fit(features_np, labels_np)
    return model

def evaluate_probe_full(probe: LogisticRegression, features: Float[Tensor, "batch d_model"], labels: Float[Tensor, "batch"]) -> tuple[float, float, float, tuple[float, float]]:
    """
    Evaluates the logistic regression probe on the given features and labels.
    Returns the loss, accuracy, and ROC AUC score.
    """
    features_np = features.to(t.float32).detach().cpu().numpy()
    labels_np = labels.to(t.float32).detach().cpu().numpy()
    
    # Get predictions and probabilities
    probas = probe.predict_proba(features_np)[:, 1]  # Probability of class 1
    preds = probe.predict(features_np)
    
    # Calculate accuracy
    accuracy = (preds == labels_np).mean()
    
    # Calculate log loss (cross-entropy loss)
    epsilon = 1e-15
    probas = np.clip(probas, epsilon, 1-epsilon)  # Avoid log(0)
    loss = -np.mean(labels_np * np.log(probas) + (1-labels_np) * np.log(1-probas))
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(labels_np, probas)
        # Get ROC curve points for potential future plotting
        fpr, tpr, _ = roc_curve(labels_np, probas)
        roc_curve_data = (fpr, tpr)
    except ValueError:
        # This can happen if there's only one class in the labels
        roc_auc = float('nan')
        roc_curve_data = None
        
    return loss, accuracy, roc_auc, roc_curve_data # type: ignore

def get_probe_weights(probe: LogisticRegression) -> Float[Tensor, "d_model"]:
    """
    Extracts weights from the logistic regression model.
    
    Args:
        probe: Trained sklearn LogisticRegression model
        activation_dim: Dimensions of the activation features
        
    Returns:
        Weight vector as a PyTorch tensor
    """
    # Sklearn's LogisticRegression stores weights in coef_ attribute
    # Shape is (n_classes, n_features) for multi-class, or (1, n_features) for binary
    weights = probe.coef_[0]  # For binary classification
    
    return t.tensor(weights, dtype=t.float32)

def train_and_evaluate_probe(train_feats: Float[Tensor, "batch d_mod_or_sae_hidden"], train_labels: Float[Tensor, "batch"], 
                            test_feats: Float[Tensor, "batch d_mod_or_sae_hidden"], test_labels: Float[Tensor, "batch"], 
                            feature_type: str, seed: int, epochs: int, batch_size: int, l1_c: float, l2_c: float
                            ) -> tuple[LogisticRegression, dict[str, float | Any], Float[Tensor, "d_mod_or_sae_hidden"]]:
    """
    Train and evaluate a logistic regression probe for a specific feature type.
    
    Args:
        train_feats: Training features
        train_labels: Training labels
        test_feats: Test features
        test_labels: Test labels
        feature_type: Type of feature ('sae_input', 'sae_diff', or 'sae_acts_post')
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        l1_c: L1 regularization strength
        l2_c: L2 regularization strength
        
    Returns:
        Tuple of (probe, results_dict, weight_vector)
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Use L1 penalty for sae_acts_post; L2 for the other probes
    penalty = 'l1' if feature_type == "sae_acts_post" else 'l2'
    C = l1_c if feature_type == "sae_acts_post" else l2_c
    
    probe = train_logistic_probe(train_feats, train_labels, 
                           epochs=epochs, batch_size=batch_size, penalty=penalty, C=C)
    
    train_loss, train_acc, train_roc_auc, _ = evaluate_probe_full(probe, train_feats, train_labels)
    test_loss, test_acc, test_roc_auc, _ = evaluate_probe_full(probe, test_feats, test_labels)
    
    weight_vector = get_probe_weights(probe)
    
    results = {
        "Seed": seed,
        "Feature_Type": feature_type,
        "Train_Loss": train_loss,
        "Train_Accuracy": train_acc,
        "Train_ROC_AUC": train_roc_auc,
        "Test_Loss": test_loss,
        "Test_Accuracy": test_acc,
        "Test_ROC_AUC": test_roc_auc,
        "Weight_Norm": t.norm(weight_vector).item()
    }
    
    return probe, results, weight_vector

def combine_sae_and_error(train_indices: NDArray[np.int32], test_indices: NDArray[np.int32], 
                            features_map: dict[str, Float[Tensor, "batch d_mod_or_sae_hidden"]], sae: SAE, 
                            device: t.device | str, top_indices: Float[Tensor, "k"]
                            ) -> tuple[Float[Tensor, "batch d_mod"], Float[Tensor, "batch d_mod"]]:
    """
    Create combined features using top-k dimensions from the sae_acts_post probe.
    
    Args:
        train_indices: Indices for training data
        test_indices: Indices for test data
        features_map: Dictionary mapping feature types to tensors
        sae: SAE model
        device: Device to use
        top_indices: Indices of top-k dimensions
        
    Returns:
        Tuple of (train_combined, test_combined)
    """
    train_acts_post = features_map["sae_acts_post"][train_indices].to(device)
    test_acts_post = features_map["sae_acts_post"][test_indices].to(device)
    train_error = features_map["sae_diff"][train_indices].to(device)
    test_error = features_map["sae_diff"][test_indices].to(device)
    
    # Multiply the filtered activations by the corresponding decoder weights using einsum notation, then add the error
    filtered_train = t.einsum('ij,jk->ik', train_acts_post[:, top_indices], sae.W_dec[top_indices, :])
    filtered_test = t.einsum('ij,jk->ik', test_acts_post[:, top_indices], sae.W_dec[top_indices, :])
    
    train_combined = filtered_train + train_error
    test_combined = filtered_test + test_error
    
    return train_combined, test_combined

def compute_cosine_similarities(weight_vectors: dict[str, Float[Tensor, "d_model"]]) -> dict[str, float]:
    """
    Compute cosine similarities between weight vectors.
    
    Args:
        weight_vectors: Dictionary mapping feature types to weight vectors
        
    Returns:
        Dictionary of cosine similarities
    """
    w_input = weight_vectors["sae_input"]
    w_error = weight_vectors["sae_diff"]
    w_combined = weight_vectors["sae_acts_and_error"]
    
    with t.no_grad():
        cos_sim_input_error = F.cosine_similarity(w_input, w_error, dim=0).item()
        cos_sim_input_combined = F.cosine_similarity(w_input, w_combined, dim=0).item()
        cos_sim_error_combined = F.cosine_similarity(w_error, w_combined, dim=0).item()
    
    return {
        "Cosine_Sim_Input_Error": cos_sim_input_error,
        "Cosine_Sim_Input_Combined": cos_sim_input_combined,
        "Cosine_Sim_Error_Combined": cos_sim_error_combined
    }

def save_probes_and_compute_similarities(models_dir: Path, results_dir: Path, save_probes_count: int, 
                                        result_suffix: str, probe_type: str | None = None, plot_dir: Path | None = None) -> None:
    """
    Compute cosine similarities for saved probes
    
    Args:
        models_dir: Directory to save models
        results_dir: Directory to save results
        plot_dir: Directory to save plots
        save_probes_count: Number of probes to save
        result_suffix: Suffix for result files
        probe_type: if it's sae_acts_post, plot the bar chart of the absolute coefficients
    Returns:
        None
    """
    if save_probes_count <= 0:
        return
        
    with t.no_grad():
        print("Computing average cosine similarities across saved sae_acts_post probes and plotting coefficient bar chart...")
        similarity_results = []
        weights_list = []
        import pickle
        
        for seed in range(save_probes_count):
            model_filename = f"probe_{probe_type}_seed_{seed}.pt"
            filepath = models_dir / model_filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    model_saved = pickle.load(f)
                weight_vector = get_probe_weights(model_saved)
                weights_list.append(weight_vector)
            else:
                warn(f"Probe file {filepath} does not exist.")
                
        sims = []
        num = len(weights_list)
        for i in range(num):
            for j in range(i+1, num):
                sim = F.cosine_similarity(weights_list[i], weights_list[j], dim=0).item()
                sims.append(sim)
                
        if sims:
            avg_sim = np.mean(sims)
            std_sim = np.std(sims)
        else:
            avg_sim = None
            std_sim = None
            
        similarity_results.append({
            "Feature_Type": probe_type,
            "Average_Cosine_Similarity": avg_sim,
            "Cosine_Similarity_Std": std_sim
        })
        
        similarities_df = pd.DataFrame(similarity_results)
        suffix = f"_{result_suffix}" if result_suffix else ""
        similarities_csv = results_dir / f"probe_similarities{suffix}.csv"
        similarities_df.to_csv(similarities_csv, index=False)

        # Plot bar chart for average absolute coefficients of sae_acts_post probe
        if probe_type == "sae_acts_post":
            assert plot_dir is not None
            stacked_weights = t.stack(weights_list)  # shape: (n_saved, feature_dim)
            avg_abs = stacked_weights.abs().mean(dim=0).cpu().numpy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,5))
            plt.bar(np.arange(len(avg_abs)), avg_abs)
            plt.xlabel("Dimension Index")
            plt.ylabel("Average Absolute Coefficient")
            plt.title("Average Absolute Coefficients for sae_acts_post Probe")
            plt.tight_layout()
            plt.savefig(plot_dir / f"sae_acts_post_coeff_bar{suffix}.png")
            plt.close()

def run_probing_pipeline(df: pd.DataFrame, device: t.device, 
                         label_col: str, features_map: dict[str, t.Tensor],
                         models_dir: Path, results_dir: Path, plot_dir: Path,
                         sae: SAE,
                         k: int = 6,
                         epochs=2, batch_size=16,
                         n_seeds=50, save_probes_count=20, 
                         l2_c=0.1, l1_c=1,
                         result_suffix=""):
    """
    Runs the complete probing pipeline with four probes:
      - Probe on the residual stream (from sae_input) [L2]
      - Probe on the reconstruction error (sae_diff) [L2]
      - Probe on the intermediate activations (sae_acts_post) [L1]
      - Combined probe (sae_acts_and_error): computed by filtering sae_acts_post via the top-k
        weight dimensions (from the L1 probe) through the decoder weights and adding sae_diff [L2]
    Also computes cosine similarities among the three d_model probes.
    Plots a bar chart of the absolute coefficients for the sae_acts_post probe.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{result_suffix}" if result_suffix else ""
    results_csv = results_dir / f"probe_results{suffix}.csv"

    results = []
    n = df.shape[0]
    train_size = int(0.8 * n)

    for seed in tqdm(range(n_seeds), desc="Training probes"):
        # Create a new random split per seed
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        labels_all = t.tensor(df[label_col].values)
        train_labels = labels_all[train_indices]
        test_labels = labels_all[test_indices]

        # Dictionaries to hold our probe objects and results
        probes_for_label = {}
        temp_results = {}
        weight_vectors = {}  # For storing weights from d_model probes
        l1_weight_vector = None

        # Train three probes on the provided features (excluding the old sae_recons)
        for feature_type in ["sae_input", "sae_diff", "sae_acts_post"]:
            feats_all = features_map[feature_type]
            train_feats = feats_all[train_indices]
            test_feats = feats_all[test_indices]
            
            probe, result, weight_vector = train_and_evaluate_probe(
                train_feats, train_labels, test_feats, test_labels,
                feature_type, seed, epochs, batch_size, l1_c, l2_c
            )
            
            
            if feature_type == 'sae_acts_post':
                l1_weight_vector = weight_vector
            elif feature_type != "sae_acts_post":
                weight_vectors[feature_type] = weight_vector
                
            probes_for_label[feature_type] = probe
            temp_results[feature_type] = result
            
            if seed < save_probes_count:
                model_filename = f"probe_{feature_type}_seed_{seed}.pt"
                with open(models_dir / model_filename, 'wb') as f:
                    pickle.dump(probe, f)

        # Create combined features (sae_acts_and_error) using top-k dimensions from the sae_acts_post probe
        # Get top k indices by absolute value from the L1 probe's weight vector
        topk = t.topk(l1_weight_vector.abs(), k)
        top_indices = topk.indices  # A tensor containing the top-k indices
        
        # Create combined features
        train_combined, test_combined = combine_sae_and_error(
            train_indices, test_indices, features_map, sae, device, top_indices
        )

        # Train combined probe
        combined_probe, combined_result, combined_weight = train_and_evaluate_probe(
            train_combined, train_labels, test_combined, test_labels,
            "sae_acts_and_error", seed, epochs, batch_size, l1_c, l2_c
        )
        if seed < save_probes_count:
            model_filename = f"probe_sae_acts_and_error_seed_{seed}.pt"
            with open(models_dir / model_filename, 'wb') as f:
                pickle.dump(combined_probe, f)
        

        combined_result["Top_k_Indices"] = top_indices.tolist()
        
        weight_vectors["sae_acts_and_error"] = combined_weight
        probes_for_label["sae_acts_and_error"] = combined_probe
        temp_results["sae_acts_and_error"] = combined_result

        # Compute cosine similarities among the three L2 probes
        similarities = compute_cosine_similarities(weight_vectors)
        
        # Add similarities to all results
        for ft in ["sae_input", "sae_diff", "sae_acts_and_error"]:
            temp_results[ft].update(similarities)

        # Append the temporary results for this seed
        for result in temp_results.values():
            results.append(result)

        t.cuda.empty_cache()

    # Save aggregated results
    results_df = pd.DataFrame(results)
    print(f"{result_suffix} finished training {n_seeds} probes.")
    print('Average Test Loss')
    print(results_df.groupby('Feature_Type')['Test_Loss'].mean())
    print('Average Test Accuracy')
    print(results_df.groupby('Feature_Type')['Test_Accuracy'].mean())
    print('Average Test ROC AUC')
    print(results_df.groupby('Feature_Type')['Test_ROC_AUC'].mean())
    
    
    better_auc = np.mean(
        results_df[results_df["Feature_Type"] == "sae_acts_and_error"]["Test_ROC_AUC"].to_numpy() > 
        results_df[results_df["Feature_Type"] == "sae_input"]["Test_ROC_AUC"].to_numpy()
    )
    print(f"Proportion of combined nodes with better ROC AUC: {better_auc:3f}")

    better_accuracy = np.mean(
        results_df[results_df["Feature_Type"] == "sae_acts_and_error"]["Test_Accuracy"].to_numpy() > 
        results_df[results_df["Feature_Type"] == "sae_input"]["Test_Accuracy"].to_numpy()
    )
    print(f"Proportion of combined nodes with better accuracy: {better_accuracy:3f}")

    results_df.to_csv(results_csv, index=False)

    # Compute similarities and save plots if needed
    return results_df

def process_dataset(dataset, result_suffix, args, model, sae, project_root, device):
    """
    Process a single dataset with the probing pipeline.
    
    Args:
        dataset: Dataset filename
        result_suffix: Suffix for result files
        args: Command line arguments
        model: Model to use
        sae: SAE to use
        project_root: Project root directory
        device: Device to use
        
    Returns:
        None
    """
    gc.collect()
    print(f"Processing dataset: {dataset}")
    
    # Load and prepare data
    df = pd.read_csv(project_root / "data" / "raw" / dataset)
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model.tokenizer)
    
    # Generate features
    print(f"Generating features for entire dataset with offset: {args.token_position_offset}")
    feats_all_input, feats_all_recons, feats_all_diff, feats_all_acts_post = generate_probing_features(
        tokenized_all, model, sae, 
        layer=args.layer,
        batch_size=args.batch_size, 
        device=device, 
        offset=args.token_position_offset
    )
    
    # Build the features map
    features_map = {
        "sae_input": feats_all_input,
        "sae_diff": feats_all_diff,
        "sae_acts_post": feats_all_acts_post
    }
    
    # Set up directories
    model_sae_id = "logistic_combined_" + args.model.replace("-", "_").replace("/", "_") + "_" + args.sae_id.replace("/", "_").replace("-", "_")
    models_dir = project_root / "models" / model_sae_id
    results_dir = project_root / "data" / "processed" / model_sae_id
    plot_dir = project_root / "reports" / "figures" / model_sae_id
    # Run probing pipeline
    _ = run_probing_pipeline(
        df=df, 
        device=device, 
        label_col=args.label_column, 
        features_map=features_map, 
        models_dir=models_dir, 
        results_dir=results_dir,
        plot_dir=plot_dir,
        sae=sae,
        k=args.k,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        n_seeds=args.n_seeds, 
        save_probes_count=args.save_probes_count,
        l1_c=args.l1_regularization,
        l2_c=args.l2_regularization,
        result_suffix=result_suffix
    )
    for probe_type in ["sae_input", "sae_diff", "sae_acts_post", "sae_acts_and_error"]:
        save_probes_and_compute_similarities(
            models_dir, results_dir, args.save_probes_count, result_suffix, probe_type, plot_dir
        )
    
    
    t.cuda.empty_cache()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train logistic regression probes on SAE activations.")
    
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
    parser.add_argument("--dataset", type=str, nargs = "+", default="all_cities.csv",
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
    parser.add_argument("--k", type=int, default=6,
                        help="Number of top activations to use for combined features")
    parser.add_argument("--l1_regularization", type=float, default=0.25,
                        help="L1 regularization strength for SAE activations probe")
    parser.add_argument("--l2_regularization", type=float, default=0.1,
                        help="L2 regularization strength for other probes")
    
    # Experiment control
    parser.add_argument("--run_patching", action="store_true",
                        help="Whether to run the patching experiment")
    parser.add_argument("--result_suffix", type=str, nargs = "+", default="truth",
                        help="Suffix for result files (e.g., 'truth', 'hl_frontp')")
    
    # Compute resources
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and inference")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    
    #Hyperparameters
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to train for")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent  # Directory of this script
    project_root = script_dir.parent    # Project root directory

    print('Setting up pipeline')
    device = t.device(args.device)
    
    # Load SAE
    print('Loading SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device="cpu"
    )
    sae = sae.to(device)
    
    # Load model
    print('Loading model')
    model = HookedSAETransformer.from_pretrained(
        args.model, 
        device='cpu', 
        dtype=t.bfloat16
    )
    model = model.to(device)

    # Ensure datasets and suffixes are lists
    if isinstance(args.dataset, str):
        args.dataset = [args.dataset]

    if isinstance(args.result_suffix, str):
        args.result_suffix = [args.result_suffix]

    assert len(args.dataset) == len(args.result_suffix), "Dataset and result suffix must have the same length"

    # Process each dataset
    for dataset, result_suffix in zip(args.dataset, args.result_suffix):
        process_dataset(
            dataset, 
            result_suffix, 
            args, 
            model, 
            sae, 
            project_root, 
            device
        )