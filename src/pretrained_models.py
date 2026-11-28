"""Extra pretrained code models for the robustness comparison (UniXcoder, CodeT5+ encoders).

Owner: Khalid Waleed Alhilal

All models are trained with the same loop as CodeBERT (src/codebert_model.train). This module only
builds, saves and loads the networks:
- RoBERTa-style models (CodeBERT, UniXcoder): AutoModelForSequenceClassification.
- CodeT5+ (encoder-decoder): only the encoder is used, followed by mean pooling over the real
  tokens and a linear layer with 2 outputs (T5EncoderClassifier).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, T5EncoderModel

KIND_FILE = "model_kind.json"


def model_kind(hf_name: str) -> str:
    """"t5enc" for CodeT5 / CodeT5+ checkpoints, "seqcls" for RoBERTa-style encoders."""
    return "t5enc" if "codet5" in hf_name.lower() else "seqcls"


class T5EncoderClassifier(nn.Module):
    """T5 encoder + mean pooling + dropout + linear layer (2 classes)."""

    def __init__(self, encoder: T5EncoderModel, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.d_model, 2)

    def forward(self, input_ids, attention_mask, labels=None, **_):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)     # mean over real tokens only
        logits = self.head(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.float(), labels)
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, out_dir) -> None:
        out = Path(out_dir)
        self.encoder.save_pretrained(out / "encoder")
        torch.save(self.head.state_dict(), out / "head.pt")

    @classmethod
    def from_saved(cls, model_dir) -> "T5EncoderClassifier":
        model = cls(T5EncoderModel.from_pretrained(Path(model_dir) / "encoder", dtype=torch.float32))
        model.head.load_state_dict(torch.load(Path(model_dir) / "head.pt", map_location="cpu"))
        return model


def load_tokenizer(name: str):
    """Tokenizer of a checkpoint or of a saved model folder.

    The CodeT5+ tokenizer config on the Hub does not load with this transformers version
    (TypeError in add_tokens), so the same byte-level BPE tokenizer is rebuilt from its
    vocab.json + merges.txt with the tokenizers library: <s> text </s>, <pad> = 0.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(name)
    except TypeError:
        from huggingface_hub import hf_hub_download
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import RobertaProcessing
        bpe = ByteLevelBPETokenizer(hf_hub_download(name, "vocab.json"), hf_hub_download(name, "merges.txt"),
                                    add_prefix_space=False)
        bpe._tokenizer.post_processor = RobertaProcessing(("</s>", bpe.token_to_id("</s>")),
                                                          ("<s>", bpe.token_to_id("<s>")))
        return PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, bos_token="<s>", eos_token="</s>",
                                       unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                       cls_token="<s>", sep_token="</s>", model_max_length=512)


def build_model(hf_name: str, grad_ckpt: bool = False) -> nn.Module:
    """New classifier on top of a pretrained checkpoint (weights in fp32; training uses bf16 autocast)."""
    if model_kind(hf_name) == "t5enc":
        model = T5EncoderClassifier(T5EncoderModel.from_pretrained(hf_name, dtype=torch.float32))
        if grad_ckpt:
            model.encoder.gradient_checkpointing_enable()
        return model
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2, attn_implementation="sdpa",
                                                               dtype=torch.float32)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def save_model(model: nn.Module, hf_name: str, out_dir) -> None:
    """Save the network and a small file that tells load_model which class to use."""
    model.save_pretrained(out_dir)
    with open(Path(out_dir) / KIND_FILE, "w", encoding="utf-8") as fh:
        json.dump({"base_model": hf_name, "kind": model_kind(hf_name)}, fh)


def load_model(model_dir) -> nn.Module:
    """Load a saved classifier (checkpoints without model_kind.json are RoBERTa-style)."""
    kind_path = Path(model_dir) / KIND_FILE
    kind = json.loads(kind_path.read_text())["kind"] if kind_path.exists() else "seqcls"
    if kind == "t5enc":
        return T5EncoderClassifier.from_saved(model_dir)
    return AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="sdpa", dtype=torch.float32)
