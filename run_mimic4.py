import sys
sys.path.append('..')
from util import generate_random_seed
from models.model import LAMRec
import argparse
from pyhealth.utils import set_seed
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import drug_recommendation_mimic4_fn
from trainer import Trainer

if __name__ == "__main__":

    # create ArgumentParser instance
    parser = argparse.ArgumentParser(description='LAMRec: Label-aware Multi-view Drug Recommendation')

    parser.add_argument('--embedding_dim', type=int, default=512, help='The dimensionality of the embedding space')
    parser.add_argument('--heads', type=int, default=4,help='The number of attention heads in the cross-attention module')
    parser.add_argument('--num_layers', type=int, default=1, help='The number of cross-attention blocks')
    parser.add_argument('--alpha', type=float, default=1e-1, help='The balancing factor for the DDI loss')
    parser.add_argument('--beta', type=float, default=1e-2,help='The balancing factor for the multi-view contrastive loss')
    parser.add_argument('--temperature', type=float, default=10,help='The balancing factor mentioned in the Eq.(13)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='The number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size for training')

    parser.add_argument('--device', type=str, default="cuda:6",help='The device to run the model on, e.g., "cuda:0" for GPU')
    parser.add_argument('--dataset_path', type=str, default="/home/lnsdu/tys/data/mimiciv", help='The dataset file path, which should contain the main csv files')
    parser.add_argument('--dev', type=bool, default=False, help='Whether to run the model in development mode')
    parser.add_argument('--refresh_cache', type=bool, default=False, help='Whether to refresh the cached dataset files')

    args = parser.parse_args()

    seed = generate_random_seed()
    set_seed(seed)

    # STEP 1: load data
    base_dataset = MIMIC4Dataset(
        root=config.dataset_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=config.dev,
        refresh_cache=config.refresh_cache,
    )
    sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn)
    sample_dataset.stat()

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # STEP 3: define model
    model = LAMRec(sample_dataset,
                    embedding_dim=args.embedding_dim,
                    heads=args.heads,
                    num_layers=args.num_layers,
                    alpha=args.alpha,
                    beta=args.beta,
                    temperature=args.temperature,
                    )

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "pr_auc_samples", "f1_samples", "ddi_score", "roc_auc_samples", "avg_med"],
        device=args.device,
        seed=seed
    )

    # train & test
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        monitor="jaccard_samples",
        lr=args.lr,
    )
