import torch
import tqdm
from copy import deepcopy
from math import sqrt
from tokenizers import normalizers, pre_tokenizers, decoders
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoConfig

class TokenizerAdapter:
    """
    Adapts an existing language model (LLM) to a new vocabulary
    by modifying the embedding and language model (LM) head layers.
    """

    def __init__(self, method="average", clean_tokenizer=False, original_model=None, custom_preprocessing=None):
        """
        Initializes the adapter.

        Args:
            method (`str`, *optional*, defaults to 'average'):
                Method for aggregating subword embeddings.
                Options: ['average', 'bos', 'frequency', 'reverse_frequency', 'inverse_frequency', 'self_attention', 'svd', 'contextual'].
            clean_tokenizer (`bool`, *optional*, defaults to False):
                Removes the normalizer, pre-tokenizer, and decoder from the old tokenizer (experimental).
            original_model (`PreTrainedModel`, *optional*, default to None):
                Required for the 'contextual' method to extract hidden states.
            custom_preprocessing (`function`, *optional*, defaults to None):
                Normalization function to apply to new tokens before passing them to the old tokenizer.
                Example: `lambda x: x.replace('Ġ', ' ')`.
        """
        valid_methods = ["average", "bos", "frequency", "reverse_frequency", "inverse_frequency", "self_attention", "svd", "contextual"]
        if method not in valid_methods:
            raise ValueError(f"Method '{method}' is not recognized. Valid methods: {valid_methods}")

        self.method = method
        self.process_function = {
            "average": self.process_average,
            "bos": self.process_bos,
            "frequency": self.process_frequency,
            "reverse_frequency": self.process_reverse_frequency,
            "inverse_frequency": self.process_inverse_frequency,
            "self_attention": self.process_self_attention_aggregation,
            "svd": self.process_svd,
            "contextual": self.process_contextual
        }[self.method]

        self.clean_tokenizer = clean_tokenizer
        self.custom_preprocessing = custom_preprocessing
        self.context_model = None # Initialize context_model

        if self.method == 'contextual':
            if original_model is None:
                raise ValueError("'original_model' must be provided to use the 'contextual' method.")
            
            # --- OPTIMIZATION: Create a shallow, 1-layer model for contextual processing ---
            print("Creating a lightweight, 1-layer model for the 'contextual' method...")
            with torch.no_grad():
                original_model.eval()
                model_class = type(original_model)
                
                # Create a new config with only one hidden layer
                shallow_config = deepcopy(original_model.config)
                shallow_config.num_hidden_layers = 1
                
                # Instantiate the shallow model structure
                self.context_model = model_class(shallow_config)
                
                # Prepare a state_dict with only the necessary weights
                original_state_dict = original_model.state_dict()
                shallow_state_dict = self.context_model.state_dict()
                
                # Copy embedding weights and the weights of the first transformer layer
                for key in shallow_state_dict.keys():
                    if key in original_state_dict:
                        shallow_state_dict[key] = original_state_dict[key].clone()
                
                self.context_model.load_state_dict(shallow_state_dict)
                self.context_model.to(original_model.device)
                self.context_model.eval()
            print("Lightweight model created successfully.")
            # We no longer need the full original model in this class
            self.original_model = None 
        else:
            self.original_model = original_model

    def get_state_dict_keys_to_update(self, state_dict, vocab_size):
        """Identifies the layers in the state_dict that have a vocabulary-sized dimension."""
        state_dict_to_update = {}
        for key, tensor in state_dict.items():
            if vocab_size in tensor.shape:
                state_dict_to_update[key] = tensor
        return state_dict_to_update

    def get_unk_token_id(self, tokenizer):
        """Finds a suitable UNK token ID, with fallbacks."""
        for token_id in [tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
            if token_id is not None:
                return token_id
        raise ValueError("Could not find an UNK, PAD, or EOS token in the old tokenizer.")

    def prepare_special_token_ids(self, correspondance_dict, new_tokenizer, old_tokenizer, unk_token_id):
        """Explicitly handles the mapping of special tokens."""
        for special_token_name in ['bos', 'eos', 'pad', 'sep', 'unk', 'mask']:
            new_id = getattr(new_tokenizer, f"{special_token_name}_token_id", None)
            old_id = getattr(old_tokenizer, f"{special_token_name}_token_id", None)
            if new_id is not None:
                correspondance_dict["pairs"][str(new_id)] = [old_id if old_id is not None else unk_token_id]
        return correspondance_dict

    def prepare_correspondance_dict(self, new_tokenizer, old_tokenizer):
        """Creates a mapping table between the new and old vocabulary IDs."""
        vocab_size = len(new_tokenizer.vocab)
        old_vocab_size = len(old_tokenizer.vocab)
        frequency_matrix = None

        unk_token_id = self.get_unk_token_id(old_tokenizer)

        if self.method in ["frequency", "reverse_frequency", "inverse_frequency"]:
            frequency_matrix = torch.zeros(old_vocab_size)

        correspondance_dict = {"pairs": {}, "meta": {}}

        print("Building the token correspondence table...")
        for new_token, i in tqdm.tqdm(new_tokenizer.vocab.items()):
            processed_token = self.custom_preprocessing(new_token) if self.custom_preprocessing else new_token

            old_token_ids = old_tokenizer.convert_tokens_to_ids([processed_token])

            if len(old_token_ids) == 0 or (len(old_token_ids) == 1 and old_token_ids[0] == unk_token_id):
                new_token_str = new_tokenizer.convert_tokens_to_string([new_token])
                old_token_ids = old_tokenizer.encode(new_token_str, add_special_tokens=False)

            old_token_ids = [t for t in old_token_ids if t < old_vocab_size]
            if not old_token_ids:
                old_token_ids = [unk_token_id]

            correspondance_dict["pairs"][str(i)] = old_token_ids

            if frequency_matrix is not None and len(old_token_ids) > 1:
                for t in old_token_ids:
                    frequency_matrix[t] += 1

        correspondance_dict = self.prepare_special_token_ids(correspondance_dict, new_tokenizer, old_tokenizer, unk_token_id)

        correspondance_dict["meta"] = {
            "vocab_size": vocab_size,
            "old_vocab_size": old_vocab_size,
            "frequency_matrix": frequency_matrix,
            "old_bos_token_id": old_tokenizer.bos_token_id,
            "bos_token_id": new_tokenizer.bos_token_id,
            "old_eos_token_id": old_tokenizer.eos_token_id,
            "eos_token_id": new_tokenizer.eos_token_id,
            "old_pad_token_id": old_tokenizer.pad_token_id,
            "pad_token_id": new_tokenizer.pad_token_id,
        }
        return correspondance_dict

    def process_tensors(self, state_dict, correspondance_dict):
        """Processes the tensors to be updated based on the correspondence dictionary."""
        vocab_size = correspondance_dict["meta"]["old_vocab_size"]

        for tensor_key, tensor in state_dict.items():
            print(f"Processing tensor: {tensor_key}")

            do_transpose = len(tensor.size()) > 1 and tensor.size()[-1] == vocab_size
            if do_transpose:
                tensor = tensor.T

            new_tensor = self.process_single_tensor(tensor, correspondance_dict)
            state_dict[tensor_key] = new_tensor.T if do_transpose else new_tensor

        return state_dict

    def process_single_tensor(self, tensor, correspondance_dict):
        """Processes a single tensor according to the chosen aggregation method."""
        new_vocab_size = correspondance_dict["meta"]["vocab_size"]
        new_tensor_shape = (new_vocab_size, tensor.shape[1]) if len(tensor.shape) > 1 else (new_vocab_size,)
        new_tensor = torch.zeros(new_tensor_shape, dtype=tensor.dtype, device=tensor.device)

        print(f"Applying method '{self.method}'...")
        for new_idx_str, old_idx_list in tqdm.tqdm(correspondance_dict["pairs"].items()):
            new_idx = int(new_idx_str)
            if not old_idx_list: continue

            value = self.process_function(old_idx_list, tensor, correspondance_dict["meta"])
            new_tensor[new_idx] = value

        return new_tensor

    def process_average(self, old_idx, tensor, meta_dict):
        """Averages the embeddings of the subwords."""
        return tensor[old_idx].mean(dim=0)

    def process_bos(self, old_idx, tensor, meta_dict):
        """Uses the beginning-of-sentence (BOS) token to weight the subword embeddings."""
        bos = tensor[meta_dict["old_bos_token_id"]]
        new_tensor_slice = tensor[old_idx]
        if len(bos.shape) == 0:
            bos = bos.unsqueeze(-1)
            new_tensor_slice = new_tensor_slice.unsqueeze(-1)

        scores = torch.softmax((bos @ new_tensor_slice.T) / sqrt(bos.size(0)), dim=-1)
        return (scores @ new_tensor_slice).squeeze(0)

    def _get_frequency_weights(self, old_idx, meta_dict, mode='normal'):
        """Calculates weights based on subword frequencies."""
        freq_matrix = meta_dict["frequency_matrix"]
        frequencies = freq_matrix / (freq_matrix.sum() + 1e-8)
        sub_frequencies = frequencies[old_idx]

        if mode == 'reverse':
            weights = 1 - sub_frequencies
        elif mode == 'inverse':
            weights = 1 / (sub_frequencies + 1e-8)
        else:
            weights = sub_frequencies

        return weights / (weights.sum() + 1e-8)

    def process_frequency(self, old_idx, tensor, meta_dict):
        """Weights subword embeddings by their frequency."""
        weights = self._get_frequency_weights(old_idx, meta_dict, mode='normal')
        sub_embeddings = tensor[old_idx]
        return torch.einsum('i,id->d', weights, sub_embeddings) if len(sub_embeddings.shape) > 1 else torch.einsum('i,i->', weights, sub_embeddings)

    def process_reverse_frequency(self, old_idx, tensor, meta_dict):
        """Weights subword embeddings by their reverse frequency."""
        weights = self._get_frequency_weights(old_idx, meta_dict, mode='reverse')
        sub_embeddings = tensor[old_idx]
        return torch.einsum('i,id->d', weights, sub_embeddings) if len(sub_embeddings.shape) > 1 else torch.einsum('i,i->', weights, sub_embeddings)

    def process_inverse_frequency(self, old_idx, tensor, meta_dict):
        """Weights subword embeddings by their inverse frequency."""
        weights = self._get_frequency_weights(old_idx, meta_dict, mode='inverse')
        sub_embeddings = tensor[old_idx]
        return torch.einsum('i,id->d', weights, sub_embeddings) if len(sub_embeddings.shape) > 1 else torch.einsum('i,i->', weights, sub_embeddings)

    def process_self_attention_aggregation(self, old_idx, tensor, meta_dict):
        """Aggregates subword embeddings using a self-attention mechanism."""
        old_embeddings = tensor[old_idx]
        if len(old_embeddings.shape) == 1: return old_embeddings.mean(dim=0)

        query = old_embeddings.mean(dim=0, keepdim=True)
        keys = values = old_embeddings

        d_k = query.shape[-1]
        scores = torch.matmul(query, keys.transpose(-2, -1)) / sqrt(d_k)
        attn_weights = torch.softmax(scores, dim=-1)

        return torch.matmul(attn_weights, values).squeeze(0)

    def process_svd(self, old_idx, tensor, meta_dict):
        """Uses Singular Value Decomposition (SVD) to find the most significant component."""
        sub_values = tensor[old_idx]
        if len(sub_values.shape) == 1: return sub_values.mean(dim=0)
        if sub_values.shape[0] < 2: return sub_values.squeeze(0)

        try:
            _, _, Vh = torch.linalg.svd(sub_values.float(), full_matrices=False)
            new_embedding = Vh[0, :].to(tensor.dtype)
        except torch.linalg.LinAlgError:
            return sub_values.mean(dim=0)

        mean_vec = sub_values.mean(dim=0)
        if torch.dot(new_embedding, mean_vec) < 0:
            new_embedding = -new_embedding
        return new_embedding

    def process_contextual(self, old_idx, tensor, meta_dict):
        """
        Uses the lightweight, 1-layer 'context_model' to get contextualized embeddings.
        """
        # If the original tensor is 1D (e.g., a bias vector), the contextual method is not applicable.
        # A contextual forward pass produces a hidden state vector (1D), which cannot be assigned to a scalar.
        # We fall back to a simple average, which is a sensible default for biases.
        if len(tensor.shape) == 1:
            return tensor[old_idx].mean(dim=0)

        # The rest of the logic is for 2D tensors (embedding matrices)
        if len(old_idx) == 1:
            return tensor[old_idx[0]]

        with torch.no_grad():
            input_ids = torch.tensor([old_idx], device=self.context_model.device)

            # Use the pre-loaded lightweight model
            outputs = self.context_model(input_ids, output_hidden_states=True)
            
            # hidden_states[0] is the initial embedding layer output
            # hidden_states[1] is the output of the first (and only) transformer layer
            first_layer_hidden_states = outputs.hidden_states[1]

            # Get the embedding of the last token
            contextual_embedding = first_layer_hidden_states[0, -1, :]

        return contextual_embedding.to(device=tensor.device, dtype=tensor.dtype)

    def prepare_new_config(self, config, new_tokenizer):
        """Creates a new model configuration based on the new tokenizer."""
        new_config = deepcopy(config)
        new_config.vocab_size = len(new_tokenizer.vocab)
        if hasattr(new_tokenizer, 'pad_token_id'): new_config.pad_token_id = new_tokenizer.pad_token_id
        if hasattr(new_tokenizer, 'bos_token_id'): new_config.bos_token_id = new_tokenizer.bos_token_id
        if hasattr(new_tokenizer, 'eos_token_id'): new_config.eos_token_id = new_tokenizer.eos_token_id
        return new_config

    def adapt_from_pretrained(self, model: PreTrainedModel, new_tokenizer: PreTrainedTokenizer, old_tokenizer: PreTrainedTokenizer):
        """
        Main function to adapt a pre-trained model to a new tokenizer.

        Args:
            model (`PreTrainedModel`):
                The pre-trained model to modify.
            new_tokenizer (`PreTrainedTokenizer`):
                The new tokenizer.
            old_tokenizer (`PreTrainedTokenizer`):
                The original tokenizer of the pre-trained model.
        """
        if self.clean_tokenizer:
            old_tokenizer._tokenizer.normalizer = normalizers.Sequence([])
            old_tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
            old_tokenizer._tokenizer.decoder = decoders.Sequence([])

        with torch.no_grad():
            state_dict = model.state_dict()
            config = model.config
            old_vocab_size = config.vocab_size

            state_dict_to_update = self.get_state_dict_keys_to_update(state_dict, old_vocab_size)
            if not state_dict_to_update:
                raise ValueError("No layers with the vocabulary dimension were found for adaptation.")

            correspondance_dict = self.prepare_correspondance_dict(new_tokenizer, old_tokenizer)

            state_dict_to_update = self.process_tensors(state_dict_to_update, correspondance_dict)

            state_dict.update(state_dict_to_update)

            new_config = self.prepare_new_config(config, new_tokenizer)

            model_class = type(model)
            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None,
                config=new_config,
                state_dict=state_dict
            )

        return new_model

# --- CORRECT USAGE EXAMPLE ---
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 1. Load the original model and tokenizer
# model_name = "meta-llama/Llama-2-7b-hf"
# original_model = AutoModelForCausalLM.from_pretrained(model_name)
# original_tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 2. Load your new tokenizer (e.g., one you have trained)
# new_tokenizer = AutoTokenizer.from_pretrained("./path/to/your/new/tokenizer")

# # 3. Initialize and use the adapter
# adapter = TokenizerAdapter(method="average", original_model=original_model)

# # Correct function call: (model, new_tokenizer, old_tokenizer)
# new_adapted_model = adapter.adapt_from_pretrained(
#     model=original_model,
#     new_tokenizer=new_tokenizer,
#     old_tokenizer=original_tokenizer
# )

# print("Adaptation complete!")
# print(f"Old vocabulary size: {original_model.config.vocab_size}")
# print(f"New vocabulary size: {new_adapted_model.config.vocab_size}")