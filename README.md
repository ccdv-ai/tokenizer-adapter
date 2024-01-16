# Tokenizer Adapter

A simple tool to adapt a pretrained Huggingface model to a new vocabulary (domain specific) with (almost) no training. \
Should work for almost all language models from the Huggingface Hub (need more test).

## Install

```
pip install tokenizer-adapter --upgrade
```

## Usage
```python
from tokenizer_adapter import TokenizerAdapter
from transformers import AutoTokenizer, AutoModelForMaskedLM

BASE_MODEL_PATH = "camembert-base"

# A simple corpus
corpus = ["A first sentence", "A second sentence", "blablabla"]

# Load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Train new vocabulary from the old tokenizer
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=300)

# Default params should work in most cases
adapter = TokenizerAdapter()

# Patch the model with the new tokenizer
model = adapter.adapt_from_pretrained(new_tokenizer, model, tokenizer)

# Save the model and the new tokenizer
model.save_pretrained("my_new_model/")
new_tokenizer.save_pretrained("my_new_model/")
```