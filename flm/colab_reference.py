"""Random-initialized dense transformer control for the Colab engineering pilot.

Shares prepared tokens, labels, optimizer, reporting and time budget with FLM.
Does not load SmolLM/Gemma weights or claim to reproduce their exact architecture.
"""
from dataclasses import dataclass, asdict
import os
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReferenceConfig:
    vocabulary: int
    hidden: int = 768
    intermediate: int = 2048
    layers: int = 23
    heads: int = 12
    kv_heads: int = 4
    architecture: str = 'random_llama_style_control'


class TransformerReference(nn.Module):
    def __init__(self, config: ReferenceConfig):
        super().__init__()
        os.environ.setdefault('USE_TF', '0')
        from transformers import LlamaConfig, LlamaForCausalLM
        self.config = config
        self.chunk = 0
        self.sparse_backend = 'none'
        c = config
        hf = LlamaConfig(vocab_size=c.vocabulary, hidden_size=c.hidden, intermediate_size=c.intermediate,
                         num_hidden_layers=c.layers, num_attention_heads=c.heads, num_key_value_heads=c.kv_heads,
                         max_position_embeddings=2048, tie_word_embeddings=True,
                         attention_dropout=0.0)
        hf._attn_implementation = 'sdpa'
        self.model = LlamaForCausalLM(hf)  # Config initialization, never from_pretrained.
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    @property
    def input(self):
        return self.model.get_input_embeddings()

    def loss_sum(self, x, labels, checkpointed=True):
        logits = self.model(x, use_cache=False).logits
        return F.cross_entropy(logits.float().reshape(-1, self.config.vocabulary), labels.reshape(-1),
                               ignore_index=-100, reduction='sum')

    def parameter_card(self):
        groups = {k: v.numel() for k, v in self.named_parameters()}
        return dict(config=asdict(self.config), trainable_parameters=sum(groups.values()), parameter_groups=groups,
                    source='Randomly initialized transformer; same data/tokenizer as Colab FLM')


def reference_config(vocabulary, target_millions):
    c = ReferenceConfig(vocabulary)
    if target_millions < 5:
        c.hidden, c.intermediate, c.heads, c.kv_heads = 96, 256, 6, 2
    per_layer = 2*c.hidden*c.hidden + 2*c.hidden*(c.hidden*c.kv_heads//c.heads) + 3*c.hidden*c.intermediate + 2*c.hidden
    c.layers = max(1, round((target_millions*1e6 - vocabulary*c.hidden - c.hidden)/per_layer))
    return c
