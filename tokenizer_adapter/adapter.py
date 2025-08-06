import torch 
import tqdm 
from copy import deepcopy
from math import sqrt
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import decoders

class TokenizerAdapter():

    def __init__(self, method="average", clean_tokenizer=False, original_model=None, custom_preprocessing=None) -> None:
        """
        Adapter an existing model with a new tokenizer

        Args:
            method (`str`, *optional*, defaults to 'average'):
                Method to use to merge tokens. In `['average', 'bos', 'frequency', 'reverse_frequency', 'inverse_frequency']`
            clean_tokenizer (`bool`, *optional*, defaults to False):
                Remove the normalizer, the pre_tokenizer and the decoder in the old tokenizer (experimental).
            original_model (`transformer`, *optional*, default to None):
                Used for 'contextual' method' to extract intermediate states.
            custom_preprocessing (`function`, *optional*, defaults to None):
                A function to apply some normalization before feeding tokens from the new vocabulary to the old tokenizer to find the ids.

                Example to replace a Llama style tokenizer by a RoBERTa style tokenizer: 
                `custom_preprocessing=lambda x: x.replace('Ġ', '▁')`
        """
        assert method in ["average", "bos", "frequency", "reverse_frequency", "inverse_frequency", "self_attention", "svd", "contextual"]
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
        self.original_model = original_model
        if self.method == 'contextual':
            assert self.original_model is not None, "'original_model' must be passed to use this 'contextual' method"
        self.custom_preprocessing = custom_preprocessing
    
    def get_state_dict_keys_to_update(self, state_dict, vocab_size):

        state_dict_to_update = {}
        for key, tensor in state_dict.items():
            if vocab_size in tensor.shape:
                state_dict_to_update[key] = tensor
        return state_dict_to_update

    def get_unk_token_id(self, old_tokenizer):

        unk_token_id = old_tokenizer.unk_token_id
        if unk_token_id is None:
            unk_token_id = old_tokenizer.pad_token_id
        if unk_token_id is None:
            unk_token_id = old_tokenizer.eos_token_id
        if unk_token_id is None:
            unk_token_id = old_tokenizer.bos_token_id
        return unk_token_id
    
    def prepare_special_token_ids(self, correspondance_dict, new_tokenizer, old_tokenizer, unk_token_id):

        if new_tokenizer.bos_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.bos_token_id)] = [
                old_tokenizer.bos_token_id if old_tokenizer.bos_token_id is not None else unk_token_id]
            
        if new_tokenizer.eos_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.eos_token_id)] = [
                old_tokenizer.eos_token_id if old_tokenizer.eos_token_id is not None else unk_token_id]
            
        if new_tokenizer.pad_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.pad_token_id)] = [
                old_tokenizer.pad_token_id if old_tokenizer.pad_token_id is not None else unk_token_id]
            
        if new_tokenizer.sep_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.sep_token_id)] = [
                old_tokenizer.sep_token_id if old_tokenizer.sep_token_id is not None else unk_token_id]
            
        if new_tokenizer.unk_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.unk_token_id)] = [
                old_tokenizer.unk_token_id if old_tokenizer.unk_token_id is not None else unk_token_id]
        
        if new_tokenizer.mask_token_id is not None:
            correspondance_dict["pairs"][str(new_tokenizer.mask_token_id)] = [
                old_tokenizer.mask_token_id if old_tokenizer.mask_token_id is not None else unk_token_id]
        
        return correspondance_dict
    
    def prepare_correspondance_dict(self, new_tokenizer, old_tokenizer):

        vocab_size = len(new_tokenizer.vocab.keys())
        old_vocab_size = len(old_tokenizer.vocab.keys())
        frequency_matrix = None 
        
        unk_token_id = self.get_unk_token_id(old_tokenizer)

        # Keep track if using 'frequency' method
        if self.method in ["frequency", "reverse_frequency", "inverse_frequency"]:
            frequency_matrix = torch.zeros(old_vocab_size)

        correspondance_dict = {"pairs": {}, "meta": {}}

        # Loop over the new vocabulary
        for new_token, i in tqdm.tqdm(new_tokenizer.vocab.items()):
            
            # Do custom preprocessing if any before to adapt to the old tokenizer
            if self.custom_preprocessing is not None:
                old_token_ids = old_tokenizer.convert_tokens_to_ids([self.custom_preprocessing(new_token)])
            else:
                # Try to find the token in the old tokenizer
                old_token_ids = old_tokenizer.convert_tokens_to_ids([new_token])

            # If token doesnt exist in old vocab
            if len(old_token_ids) == 0 or (len(old_token_ids) == 1 and old_token_ids[0] == unk_token_id):
                # Detokenize new_token
                new_token = new_tokenizer.convert_tokens_to_string([new_token])
            
                # Get old ids
                old_token_ids = old_tokenizer.encode(new_token, add_special_tokens=False)
                
            # Remove unk ids
            old_token_ids = [t if t < old_vocab_size else unk_token_id for t in old_token_ids]
            if len(old_token_ids) == 0:
                old_token_ids = [unk_token_id]

            # Add pair
            correspondance_dict["pairs"][str(i)] = old_token_ids

            # Fill frequency matrix
            if frequency_matrix is not None and len(old_token_ids) > 1:
                for t in old_token_ids:
                    frequency_matrix[t] += 1

        # Process special tokens
        correspondance_dict = self.prepare_special_token_ids(correspondance_dict, new_tokenizer, old_tokenizer, unk_token_id)

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

    def process_self_attention_aggregation(self, old_idx, tensor, meta_dict):
        
        old_embeddings = tensor[old_idx]
        
        # Query: la moyenne des embeddings des sous-mots
        query = old_embeddings.mean(dim=0, keepdim=True)
        
        # Keys & Values: les embeddings des sous-mots eux-mêmes
        keys = values = old_embeddings
        
        # Calcul de l'attention
        d_k = query.shape[-1]
        scores = torch.matmul(query, keys.transpose(-2, -1)) / sqrt(d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Appliquer les poids pour obtenir le nouvel embedding
        new_tensor = torch.matmul(attn_weights, values).squeeze(0)
        
        return new_tensor

    def process_svd(self, old_idx, tensor, meta_dict):
        
        if len(old_idx) == 1:
            return tensor[old_idx[0]]
    
        # Empiler les embeddings des sous-tokens
        sub_embeddings = tensor[old_idx]
        
        # S'assurer qu'il y a assez de vecteurs pour la SVD
        if sub_embeddings.shape[0] < 2:
            return sub_embeddings.mean(dim=0)
    
        # Calculer la SVD
        U, S, Vh = torch.linalg.svd(sub_embeddings, full_matrices=False)
        
        # Le nouveau vecteur est le premier composant principal (première colonne de Vh)
        # mis à l'échelle par la première valeur singulière et la moyenne des normes.
        # Une heuristique simple est de prendre la première colonne de U * la première valeur singulière
        new_tensor = U[:, 0] * S[0]
        
        return new_tensor

    def process_contextual(self, old_idx, tensor, meta_dict):
        
        # tensor est ici la matrice d'embedding statique, mais nous ne l'utiliserons que
        # si la séquence est de longueur 1.
        if len(old_idx) == 1:
            return tensor[old_idx[0]]
    
        with torch.no_grad():
            self.original_model.eval()
            
            input_ids = torch.tensor([old_idx], device=self.original_model.device)
            
            # Accès aux couches du modèle. Cela peut varier.
            # Pour RoBERTa: self.original_model.roberta.encoder.layer[0]
            # Pour BERT: self.original_model.bert.encoder.layer[0]
            # Nous allons supposer une structure générique pour l'exemple.
            
            # 1. Obtenir les embeddings statiques
            embeddings = self.original_model.get_input_embeddings()(input_ids)
            
            # 2. Passer à travers la première couche d'attention
            # L'accès exact peut nécessiter d'inspecter l'architecture de votre modèle.
            # Par exemple, pour un modèle Hugging Face standard :
            encoder_layers = self.original_model.base_model.encoder.layer
            first_layer_output = encoder_layers[0](embeddings)[0] # [0] pour obtenir les hidden_states
    
            # 3. Stratégie : prendre l'état caché du dernier sous-token
            contextual_embedding = first_layer_output[0, -1, :]
            
        # S'assurer que le vecteur est sur le bon device (CPU/GPU) et a le bon type
        return contextual_embedding.to(device=tensor.device, dtype=tensor.dtype)

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

        Returns: `PreTrainedModel`
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
        
