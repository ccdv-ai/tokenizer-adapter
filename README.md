# Tokenizer Adapter

A simple tool for adapting a pre-trained Huggingface model to a new vocabulary with (almost) no training. \
Should work for most Huggingface Hub language models (requires further testing). \
**Everything is run on CPU.**

## Install

```
pip install tokenizer-adapter --upgrade
```

## Usage
It is recommended to use an existing tokenizer to train the new vocabulary (`tokenizer.train_new_from_iterator(...)`).

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