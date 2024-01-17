# Tokenizer Adapter

A simple tool for adapting a pretrained Huggingface model to a new vocabulary with (almost) no training.

This technique can significantly reduce sequence length when a language model is used on data with a specific vocabulary (biology, medicine, law, other languages, etc...). 

A slight loss of model quality is likely to be observed, especially when the vocabulary size is greatly reduced. Fine-tuning or additional pretraining during few steps solves the problem in most cases.

Should work for most Huggingface Hub language models (requires further testing). \
**Everything is run on CPU.**

## Install

```
pip install tokenizer-adapter --upgrade
```

## Usage
It is recommended to use an existing tokenizer to train the new vocabulary. \
Best and easiest way is to use the `tokenizer.train_new_from_iterator(...)` method.

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

To rely on a custom tokenizer (**experimental**), you may need to use the `custom_preprocessing` argument. \
Example using a RoBERTa (similar to Phi-2) style tokenizer for a CamemBERT model:

```python
from tokenizer_adapter import TokenizerAdapter
from transformers import AutoTokenizer, AutoModelForMaskedLM

BASE_MODEL_PATH = "camembert-base"
NEW_CUSTOM_TOKENIZER = "roberta-base"

# A simple corpus
corpus = ["A first sentence", "A second sentence", "blablabla"]

# Load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Also load this custom tokenizer to train the new one
new_tokenizer = AutoTokenizer.from_pretrained(NEW_CUSTOM_TOKENIZER)
new_tokenizer = new_tokenizer.train_new_from_iterator(corpus, vocab_size=300)

# CamemBERT tokenizer relies on '▁' while the RoBERTa one relies on 'Ġ'
adapter = TokenizerAdapter(custom_preprocessing=lambda x: x.replace('Ġ', '▁'))

# Patch the model with the new tokenizer
model = adapter.adapt_from_pretrained(new_tokenizer, model, tokenizer)

# Save the model and the new tokenizer
model.save_pretrained("my_new_model/")
new_tokenizer.save_pretrained("my_new_model/")
```