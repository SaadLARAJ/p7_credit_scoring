from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les extraits fictifs et journalise les dimensions."""
    clients = pd.read_csv(DATA_DIR / "clients_sample.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions_sample.csv")
    products = pd.read_csv(DATA_DIR / "products_sample.csv")
    print(
        f"Loaded {len(clients)} clients, {len(transactions)} transactions, {len(products)} products",
    )
    return clients, transactions, products


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Construit quelques agrégats temporels simulant les tables Kaggle."""
    return (
        transactions.groupby("client_id")
        .agg(
            n_transactions=("transaction_id", "count"),
            total_spent=("amount", "sum"),
            avg_ticket=("amount", "mean"),
            days_since_last=("days_since", "min"),
        )
        .reset_index()
    )


def engineer_product_mix(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Joint les produits pour exposer les taux d'intérêt et la catégorie."""
    merged = transactions.merge(products, on="product_id", how="left")
    pivot = (
        merged.pivot_table(
            index="client_id",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .add_prefix("spent_")
        .reset_index()
    )
    interest = merged.groupby("client_id")["interest_rate"].mean().reset_index(name="avg_interest_rate")
    tenor = merged.groupby("client_id")["tenor_months"].max().reset_index(name="max_tenor")
    return pivot.merge(interest, on="client_id", how="left").merge(tenor, on="client_id", how="left")


def assemble_dataset() -> pd.DataFrame:
    """Pipeline d'assemblage complet + sauvegarde CSV."""
    clients, transactions, products = load_sources()
    tx_summary = aggregate_transactions(transactions)
    product_mix = engineer_product_mix(transactions, products)
    dataset = (
        clients.merge(tx_summary, on="client_id", how="left")
        .merge(product_mix, on="client_id", how="left")
        .fillna({"n_transactions": 0, "total_spent": 0, "avg_ticket": 0, "days_since_last": 999})
    )
    output_path = OUTPUT_DIR / "joined_clients.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    return dataset


if __name__ == "__main__":
    assemble_dataset()
