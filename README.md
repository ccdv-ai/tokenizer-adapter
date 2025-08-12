# Tokenizer Adapter

A simple tool to adapt a pretrained Hugging Face transformer model to a new, specialized vocabulary with minimal to no retraining.

This technique can significantly reduce sequence length and computational cost when applying a general-purpose language model to domain-specific data, such as in biology, medicine, law, or other languages.

While a slight decrease in accuracy might be observed, especially with a significantly smaller vocabulary, this can typically be mitigated with a few steps of fine-tuning or further pre-training.

The library is designed to work with most language models available on the Hugging Face Hub and runs on the CPU by default.

## Why Use Tokenizer Adapter?

Pretrained language models from the Hugging Face Hub, like `roberta-base` or `modernbert`, are trained on vast, general-domain text. Their tokenizers are optimized for this general vocabulary. When run these models on a specific domain (e.g., legal documents, scientific papers), words are often oversplitted. This leads to:

*   **Longer input sequences:** This increases memory consumption and computational time.
*   **Potential loss of semantic meaning:** Sub-optimal tokenization can obscure the meaning of domain-specific terms.

**Tokenizer Adapter** solves this by reconfiguring the model's token embeddings to match a new, more efficient tokenizer trained on your target corpus. This results in shorter sequences, faster processing, and potentially better performance after fine-tuning.

## Installation

Install the package using pip:

```bash
pip install tokenizer-adapter --upgrade
```

## How It Works

The core idea is to create a new, specialized vocabulary and tokenizer from your target corpus. Then, the `TokenizerAdapter` maps the embeddings of the original model's vocabulary to the new vocabulary. This is achieved by leveraging different methods to approximate the embeddings for the new tokens based on the existing ones.

The library offers several methods for this adaptation, including:

*   `'average'`: Averages the embeddings of the old tokens that constitute a new token. (Recommended)
*   `'first_attention'`: Uses the embedding of the first sub-token based on attention scores.
*   And many others like `'bos_attention'`, `'self_attention'`, `'frequency'`, `'svd'`, etc.

## Basic Usage

The most straightforward approach is to train a new tokenizer from your corpus using the `train_new_from_iterator()` method of an existing tokenizer.

```python
from tokenizer_adapter import TokenizerAdapter
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. Define paths and parameters
BASE_MODEL_PATH = "roberta-base"
SAVE_MODEL_PATH = "my-adapted-roberta"
VOCAB_SIZE = 5000  # Adjust based on your corpus size and domain specificity

# 2. Prepare your corpus (a list of strings)
# For a real-world scenario, this would be a large dataset
corpus = [
    "This is a sentence from my domain-specific corpus.",
    "It contains specialized terminology that the base model may not handle well.",
    # ... more sentences
]

# 3. Load the base model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# To run on a GPU, uncomment the following line:
# model.cuda()

# 4. Train a new tokenizer on your corpus
# This new tokenizer will be optimized for your data
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=VOCAB_SIZE)

# 5. Initialize the adapter and adapt the model
# The 'average' method is a robust default choice
adapter = TokenizerAdapter(method="average")
model = adapter.adapt_from_pretrained(model, new_tokenizer, tokenizer)

# 6. Save your new, adapted model and tokenizer
model.save_pretrained(SAVE_MODEL_PATH)
new_tokenizer.save_pretrained(SAVE_MODEL_PATH)

print(f"Model and tokenizer adapted and saved to {SAVE_MODEL_PATH}")
```

## Advanced Usage: Custom Tokenizer (Experimental)

In some cases, you might want to use a tokenizer with a different architecture (e.g., adapting a CamemBERT model with a RoBERTa-style tokenizer). This is an experimental feature that may require a `custom_preprocessing` function to align the token representations.

For instance, CamemBERT uses `▁` as a prefix for sub-word units, while RoBERTa uses `Ġ`. The `custom_preprocessing` function can handle such differences.

```python
from tokenizer_adapter import TokenizerAdapter
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. Define paths and parameters
BASE_MODEL_PATH = "camembert-base"
NEW_CUSTOM_TOKENIZER_PATH = "roberta-base"
SAVE_MODEL_PATH = "my-adapted-camembert"
VOCAB_SIZE = 5000

# 2. Prepare your corpus
corpus = [
    "Une phrase d'exemple pour notre corpus.",
    "Avec une terminologie spécifique au domaine.",
    # ...
]

# 3. Load the base model and its original tokenizer
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH)
original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 4. Load the custom tokenizer and train it on your corpus
custom_tokenizer_template = AutoTokenizer.from_pretrained(NEW_CUSTOM_TOKENIZER_PATH)
new_tokenizer = custom_tokenizer_template.train_new_from_iterator(corpus, vocab_size=VOCAB_SIZE)

# 5. Define a preprocessing function to handle tokenization differences
# CamemBERT's ' ' vs. RoBERTa's 'Ġ'
def roberta_to_camembert_preprocessing(token):
    return token.replace('Ġ', '▁')

# 6. Initialize the adapter with the custom preprocessing and adapt the model
adapter = TokenizerAdapter(custom_preprocessing=roberta_to_camembert_preprocessing)
model = adapter.adapt_from_pretrained(model, new_tokenizer, original_tokenizer)

# 7. Save your adapted model and new tokenizer
model.save_pretrained(SAVE_MODEL_PATH)
new_tokenizer.save_pretrained(SAVE_MODEL_PATH)

print(f"Model with custom tokenizer adapted and saved to {SAVE_MODEL_PATH}")
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/ccdv-ai/tokenizer-adapter).

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](https://github.com/ccdv-ai/tokenizer-adapter/blob/main/LICENSE) file for details.