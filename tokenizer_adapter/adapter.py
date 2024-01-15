import torch 
import tqdm 
from copy import deepcopy
from math import sqrt
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import decoders

class TokenizerAdapter():

    def __init__(self, method="average", clean_tokenizer=False, custom_preprocessing=None) -> None:
        """
        Adapter an existing model with a new tokenizer
        Args:
            method (`str`, *optional*, defaults to 'average'):
                Method to use to merge tokens. In ["average", "bos", "frequency", "reverse_frequency", "inverse_frequency"]
            clean_tokenizer (`bool`, *optional*, defaults to False):
                Remove the normalizer, the pre_tokenizer and the decoder in the old tokenizer (experimental)
            custom_preprocessing (`function`, *optional*, defaults to None):
                A custom function to apply some normalization before feeding tokens from the new vocabulary to the old tokenizer.
                Example replacing a metaspace by a RoBERTa separator: lambda x: x.replace("▁", "Ġ")
        """
        assert method in ["average", "bos", "frequency", "reverse_frequency", "inverse_frequency"]
        self.method = method
        self.process_function = {
            "average": self.process_average, 
            "bos": self.process_bos,
            "frequency": self.process_frequency,
            "reverse_frequency": self.process_reverse_frequency,
            "inverse_frequency": self.process_inverse_frequency
            }[self.method]
        self.clean_tokenizer = clean_tokenizer
    
    def get_state_dict_keys_to_update(self, state_dict, vocab_size):

        state_dict_to_update = {}
        for key, tensor in state_dict.items():
            if vocab_size in tensor.shape:
                state_dict_to_update[key] = tensor
        return state_dict_to_update

    def prepare_correspondance_dict(self, new_tokenizer, old_tokenizer):

        vocab_size = len(new_tokenizer.vocab.keys())
        old_vocab_size = len(old_tokenizer.vocab.keys())
        frequency_matrix = None 
        
        unk_token_id = old_tokenizer.unk_token_id
        if unk_token_id is None:
            unk_token_id = old_tokenizer.pad_token_id
        if unk_token_id is None:
            unk_token_id = old_tokenizer.eos_token_id

        if self.method in ["frequency", "reverse_frequency", "inverse_frequency"]:
            frequency_matrix = torch.zeros(old_vocab_size)

        correspondance_dict = {"pairs": {}, "meta": {}}

        # Loop over the new vocabulary
        for new_token, i in tqdm.tqdm(new_tokenizer.vocab.items()):

            old_token_ids = old_tokenizer.convert_tokens_to_ids([new_token])
            # if token doesnt exist in old vocab
            if len(old_token_ids) == 0 or (len(old_token_ids) == 1 and old_token_ids[0] == unk_token_id):
                # untokenize new_token
                new_token = new_tokenizer.convert_tokens_to_string([new_token])
                old_token_ids = old_tokenizer.encode(new_token, add_special_tokens=False)
            
            old_token_ids = [t if t < old_vocab_size else unk_token_id for t in old_token_ids]
            correspondance_dict["pairs"][str(i)] = old_token_ids

            # Fill frequency matrix
            if frequency_matrix is not None and len(old_token_ids) > 1:
                for t in old_token_ids:
                    frequency_matrix[t] += 1

        correspondance_dict["meta"]["vocab_size"] = vocab_size
        correspondance_dict["meta"]["old_vocab_size"] = old_vocab_size
        correspondance_dict["meta"]["frequency_matrix"] = frequency_matrix

        correspondance_dict["meta"]["old_bos_token_id"] = old_tokenizer.bos_token_id
        correspondance_dict["meta"]["bos_token_id"] = new_tokenizer.bos_token_id
        correspondance_dict["meta"]["old_eos_token_id"] = old_tokenizer.eos_token_id
        correspondance_dict["meta"]["eos_token_id"] = new_tokenizer.eos_token_id
        correspondance_dict["meta"]["old_pad_token_id"] = old_tokenizer.pad_token_id
        correspondance_dict["meta"]["pad_token_id"] = new_tokenizer.pad_token_id

        return correspondance_dict

    def process_tensors(self, state_dict, correspondance_dict):
        vocab_size = correspondance_dict["meta"]["old_vocab_size"]

        for tensor_key, tensor in state_dict.items():
            
            print("Processing: ", tensor_key)
            do_transpose = False
            if len(tensor.size()) > 1 and tensor.size()[-1] == vocab_size:
                do_transpose = True
                tensor = tensor.T

            new_tensor = self.process_single_tensor(tensor, correspondance_dict)
            state_dict[tensor_key] = new_tensor.T if do_transpose else new_tensor
        
        return state_dict
    
    def process_single_tensor(self, tensor, correspondance_dict):

        vocab_size = correspondance_dict["meta"]["vocab_size"]

        if len(tensor.size()) > 1:
            new_tensor = torch.zeros(vocab_size, tensor.size()[-1], dtype=tensor.dtype)
        else:
            new_tensor = torch.zeros(vocab_size, dtype=tensor.dtype)

        for new_idx, old_idx in tqdm.tqdm(correspondance_dict["pairs"].items()):
            new_idx = int(new_idx)
            value = self.process_function(old_idx, tensor, correspondance_dict["meta"])
            new_tensor[new_idx] = value
        return new_tensor

    def process_average(self, old_idx, tensor, meta_dict):
        new_tensor = tensor[old_idx].mean(dim=0)
        return new_tensor
    
    def process_bos(self, old_idx, tensor, meta_dict):

        bos = tensor[meta_dict["old_bos_token_id"]]
        new_tensor = tensor[old_idx]
        if len(bos.size()) == 0:
            bos = bos.unsqueeze(-1)
            new_tensor = new_tensor.unsqueeze(-1)
        new_tensor = torch.softmax(bos @ new_tensor.T / sqrt(bos.size()[0]), dim=-1) @ new_tensor
        return new_tensor
    
    def process_frequency(self, old_idx, tensor, meta_dict):
        
        frequencies = meta_dict["frequency_matrix"] / meta_dict["frequency_matrix"].sum() + 1e-8
        frequencies = frequencies[old_idx]
        frequencies = frequencies / frequencies.sum()
        new_tensor = tensor[old_idx]

        if len(new_tensor.size()) == 1:
            new_tensor = new_tensor.unsqueeze(-1)
        new_tensor = (new_tensor * frequencies.unsqueeze(-1)).sum(dim=0)
        return new_tensor
    
    def process_reverse_frequency(self, old_idx, tensor, meta_dict):
        
        frequencies = meta_dict["frequency_matrix"] / meta_dict["frequency_matrix"].sum() + 1e-8
        frequencies = 1 - frequencies[old_idx]
        frequencies = frequencies / frequencies.sum()
        new_tensor = tensor[old_idx]

        if len(new_tensor.size()) == 1:
            new_tensor = new_tensor.unsqueeze(-1)
        new_tensor = (new_tensor * frequencies.unsqueeze(-1)).sum(dim=0)
        return new_tensor
    
    def process_inverse_frequency(self, old_idx, tensor, meta_dict):
        
        frequencies = meta_dict["frequency_matrix"] / meta_dict["frequency_matrix"].sum() + 1e-8
        frequencies = 1 / frequencies[old_idx]
        frequencies = frequencies / frequencies.sum()
        new_tensor = tensor[old_idx]

        if len(new_tensor.size()) == 1:
            new_tensor = new_tensor.unsqueeze(-1)
        new_tensor = (new_tensor * frequencies.unsqueeze(-1)).sum(dim=0)
        return new_tensor

    def merge_dict(self, state_dict, state_dict_keys_updated):

        for key, value in state_dict_keys_updated.items():
            requires_grad = state_dict[key].requires_grad
            state_dict[key] = value
        return state_dict
    
    def prepare_new_config(self, config, correspondance_dict):   
        config.pad_token_id = correspondance_dict["meta"]["pad_token_id"]
        config.bos_token_id = correspondance_dict["meta"]["bos_token_id"]
        config.eos_token_id = correspondance_dict["meta"]["eos_token_id"]
        config.vocab_size = correspondance_dict["meta"]["vocab_size"]
        return config

    def adapt_from_pretrained(self, new_tokenizer, model, tokenizer, **kwargs):
        
        """
        Adapt a new model from a pretrained model and a pretrained tokenizer
        Args:
            new_tokenizer (`PreTrainedTokenizer`):
                The new tokenizer trained on a specific corpus
            model (`PreTrainedModel`):
                The pretrained model to modify
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer of the pretrained model
        """

        if self.clean_tokenizer:
            tokenizer._tokenizer.normalizer = normalizers.Sequence([])
            tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
            tokenizer._tokenizer.decoder = decoders.Sequence([])

        with torch.no_grad():
            #state_dict = deepcopy(model.state_dict())
            state_dict = model.state_dict()
            config = deepcopy(model.config)
            config_vocab_size = model.config.vocab_size
            #del model

            # Select keys to update
            state_dict_keys_to_update = self.get_state_dict_keys_to_update(state_dict, config_vocab_size)
            if len(state_dict_keys_to_update.keys()) == 0:
                state_dict_keys_to_update = self.get_state_dict_keys_to_update(state_dict, len(tokenizer.vocab.keys()))

            # Create correspondance table
            correspondance_dict = self.prepare_correspondance_dict(new_tokenizer, tokenizer)

            # Update config
            config = self.prepare_new_config(config, correspondance_dict)
            # Update tensors
            state_dict_keys_to_update = self.process_tensors(state_dict_keys_to_update, correspondance_dict)
            # Merge in state dict
            state_dict = self.merge_dict(state_dict, state_dict_keys_to_update)

            model = model.from_pretrained(pretrained_model_name_or_path=None, config=config, state_dict=state_dict)

        return model
        
