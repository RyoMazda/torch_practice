import logging
from typing import Optional

import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)


class MyBertModel(torch.nn.Module):  # type: ignore
    """Almost same as
    transformers.models.bert.modeling_bert.BertForSequenceClassification
    """
    def __init__(
            self,
            output_dim: int,
            tokenizer: transformers.BertJapaneseTokenizer,
            bert_model: transformers.BertModel,
            padding_id: int = 0,
            max_length: int = 512,
            emb_size: int = 768,
            dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self._set_fine_tuning_weights()
        self.padding_id = padding_id
        self.max_length = max_length
        self.dropout = torch.nn.Dropout(dropout_rate)  # type: ignore
        self.linear = torch.nn.Linear(emb_size, output_dim)  # type: ignore
        logger.info("Parameters:")
        logger.info("requires_grad\tname\tshape")
        for name, param in self.named_parameters():
            logger.info(f"{param.requires_grad}\t{name}\t{tuple(param.data.shape)}")

    def _set_fine_tuning_weights(self) -> None:
        for name, param in self.bert_model.named_parameters():
            if (
                    name.find('pooler') == -1
                    and name.find('layer.11') == -1
            ):
                param.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,  # (batch, seq)
            token_type_ids: Optional[torch.Tensor] = None,  # (batch, seq)
            attention_mask: Optional[torch.Tensor] = None,  # (batch, seq)
    ) -> torch.Tensor:  # (batch, num_classes)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).to(input_ids.device)  # type: ignore
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_id).to(torch.float32).to(input_ids.device)  # type: ignore
        x = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).pooler_output
        # (batch, emb_size)
        x = self.dropout(x)
        x = self.linear(x)  # (batch, output_dim)
        if x.shape[1] == 1:
            x = x.squeeze()
        return x  # type: ignore

    def predict(
            self,
            text: str,
            device: torch.device = torch.device('cpu'),  # type: ignore
    ) -> np.ndarray:  # (output_dim)
        """Given a text, returns
            fIf output_dim == 1: sigmoid(logit)
            else: softmax(logits)
        as a numpy array of size (output_dim).
        """
        x = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.eval()
        with torch.no_grad():
            x = self.forward(
                input_ids=x['input_ids'].to(device),
                token_type_ids=x['token_type_ids'].to(device),
                attention_mask=x['attention_mask'].to(device),
            )  # (1, output_dim)
            x = x.squeeze(dim=0)
            if x.size().numel() == 1:
                x = torch.sigmoid(x)  # type: ignore
            else:
                x = torch.softmax(x, dim=0)  # type: ignore
        return x.cpu().detach().numpy()
