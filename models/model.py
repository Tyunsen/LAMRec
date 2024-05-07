import os
from typing import List, Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from models.layers import TransformerCrossAttn, LabelAttention, MultiViewContrastiveLoss
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
import torch.nn.functional as F

class LAMRec(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        super(LAMRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()

        self.conditions_embeddings = nn.Embedding(
            self.feat_tokenizers['conditions'].get_vocabulary_size(),
            embedding_dim,
            padding_idx=self.feat_tokenizers['conditions'].get_padding_index(),
        )
        self.procedures_embeddings = nn.Embedding(
            self.feat_tokenizers['procedures'].get_vocabulary_size(),
            embedding_dim,
            padding_idx=self.feat_tokenizers['procedures'].get_padding_index(),
        )

        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.seq_encoder = TransformerCrossAttn(d_model=embedding_dim, nhead=kwargs['heads'],
                                                num_layers=kwargs['num_layers'], dim_feedforward=embedding_dim)
        self.multi_view_cl = MultiViewContrastiveLoss(temperature=kwargs['temperature'])
        self.label_wise_attention = LabelAttention(embedding_dim * 2, embedding_dim,
                                                   self.label_tokenizer.get_vocabulary_size())

        self.ddi_adj = self.generate_ddi_adj().to(self.device)
        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj)

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def forward(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)

        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        conditions = self.conditions_embeddings(conditions)
        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        procedures = self.procedures_embeddings(procedures)
        procedures = torch.sum(procedures, dim=2)

        diag_out, proc_out = self.seq_encoder(conditions, procedures, mask)

        mvcl = self.multi_view_cl(diag_out, proc_out)

        patient_rep = torch.cat((diag_out, proc_out), dim=-1)
        logits = self.label_wise_attention(patient_rep)

        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)

        loss = F.binary_cross_entropy_with_logits(logits, curr_drugs)
        y_prob = torch.sigmoid(logits)

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        current_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())

        if current_ddi_rate >= 0.06:
            mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
            batch_ddi_loss = (
                    torch.sum(mul_pred_prob.mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2
            )
            loss += self.alpha * batch_ddi_loss

        return {
            "loss": loss + self.beta * mvcl,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }
