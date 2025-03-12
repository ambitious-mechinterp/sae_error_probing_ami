import torch as t
from torch import Tensor
from transformers import PreTrainedTokenizerFast  # type: ignore
import pandas as pd
import numpy as np
from jaxtyping import Float


###############################################################################
# Data Helpers
###############################################################################
def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 123
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates train test split for a whole dataframe"""
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int((1 - test_size) * len(shuffled))
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]


def tokenize_data(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, text_column: str = "prompt"
) -> dict[str, t.Tensor]:
    """Tokenizes the text_column of a dataframe"""
    texts = df[text_column].tolist()
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    return tokenized


###############################################################################
# Last Token Extraction Helpers
###############################################################################
def get_last_token_indices(attention_mask: t.Tensor, offset: int = 1) -> t.Tensor:
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


def extract_last_token_acts(
    act_tensor: Float[Tensor, "batch seq_len dim"],
    attention_mask: Float[Tensor, "batch seq_len"],
    offset: int = 1,
) -> Float[Tensor, "batch dim"]:
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
