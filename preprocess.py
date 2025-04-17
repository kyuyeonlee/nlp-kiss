import pandas as pd
import re
import torch
from typing import Tuple, List, Set

def normalize_korean(text: str) -> str:
    """
    Normalize Korean text by collapsing multiple spaces and normalizing quotes.
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces.
    text = re.sub(r'[“”‘’]', '"', text)  # Normalize quotes.
    return text.strip()

def generate_spacing_data(sentence: str) -> Tuple[str, List[int], Set[str]]:
    """
    Convert a sentence into its unspaced form and generate binary spacing labels.
    
    Returns:
        - unspaced text: the sentence without spaces.
        - labels: list of binary values indicating whether a space should follow each character.
        - encountered_chars: set of characters found in the sentence.
    """
    sentence = normalize_korean(sentence)
    unspaced_text = ""
    labels = []
    encountered_chars = set()
    
    # Process each character and generate label based on following space.
    for i, char in enumerate(sentence):
        if char == " ":
            continue
        encountered_chars.add(char)
        unspaced_text += char
        if i + 1 < len(sentence) and sentence[i + 1] == " ":
            labels.append(1)
        else:
            labels.append(0)
    
    return unspaced_text, labels, encountered_chars

def generate_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Generate label data for sentences contained in a DataFrame.
    This refactored version avoids using .apply() to reduce memory usage by iterating row by row.
    
    It adds two new columns:
        - 'unspaced': sentence with spaces removed.
        - 'labels': binary labels indicating where spaces should be inserted.
    
    Returns:
        - A new DataFrame with the additional columns.
        - A set of all characters encountered across all sentences.
    """
    all_chars = set()
    unspaced_list = []
    labels_list = []

    # Iterate directly over the sentences.
    for sentence in df['sentence']:
        unspaced, labels, chars = generate_spacing_data(sentence)
        unspaced_list.append(unspaced)
        labels_list.append(labels)
        all_chars.update(chars)

    # Create a copy of the DataFrame and assign the new columns.
    df = df.copy()
    df['unspaced'] = unspaced_list
    df['labels'] = labels_list

    return df, all_chars

def generate_mappings(chars: Set[str]) -> Tuple[dict, List[str]]:
    """
    Create character-to-index and index-to-character mappings.
    A reserved token '<PAD>' is added at index 0.
    
    Args:
        chars: Set of characters.
        
    Returns:
        - char2idx: mapping from character to index.
        - idx2char: list of characters, where the list index corresponds to the mapping.
    """
    char2idx = {"<PAD>": 0}
    idx2char = ["<PAD>"]

    # Sort to ensure a consistent ordering.
    for idx, char in enumerate(sorted(chars), start=1):
        char2idx[char] = idx
        idx2char.append(char)

    return char2idx, idx2char

def generate_tensors(df: pd.DataFrame, char2idx: dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert text and label sequences from a DataFrame into tensors.
    
    Args:
        df: DataFrame containing the 'unspaced' text and 'labels'.
        char2idx: Mapping of characters to their numeric indices.
        
    Returns:
        - input_sequences: List of tensor sequences corresponding to the text.
        - label_sequences: List of tensor sequences corresponding to the labels.
    """
    input_sequences = []
    label_sequences = []

    for _, row in df.iterrows():
        # Map each character in the unspaced text to its index.
        char_indices = [char2idx[char] for char in row['unspaced']]
        input_sequences.append(torch.tensor(char_indices, dtype=torch.long))
        label_sequences.append(torch.tensor(row['labels'], dtype=torch.long))

    return input_sequences, label_sequences