from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from create_data import create_demo_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.xlsx"
TRANSACTIONS_FILE = DATA_DIR / "transactions.xlsx"


def ensure_demo_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists() or not TRANSACTIONS_FILE.exists():
        create_demo_data(DATA_DIR)


def load_users() -> pd.DataFrame:
    ensure_demo_data()
    return pd.read_excel(USERS_FILE)


def load_transactions() -> pd.DataFrame:
    ensure_demo_data()
    return pd.read_excel(TRANSACTIONS_FILE)


def get_user_by_username(username: str) -> Dict[str, Any] | None:
    df = load_users()
    match = df[df["username"].astype(str).str.lower() == username.strip().lower()]
    if match.empty:
        return None
    user = match.iloc[0].to_dict()
    return user


def get_user_by_id(user_id: str | int) -> Dict[str, Any] | None:
    df = load_users()
    match = df[df["user_id"].astype(str).str.strip() == str(user_id).strip()]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_safe_account(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": user.get("name"),
        "account_number": user.get("account_number"),
        "account_type": user.get("account_type"),
        "balance": int(float(user.get("balance", 0))),
        "branch": user.get("branch"),
        "ifsc": user.get("ifsc"),
        "status": user.get("account_status"),
    }


def get_user_transactions(user_id: str | int) -> List[Dict[str, Any]]:
    df = load_transactions()
    filtered = df[df["user_id"].astype(str).str.strip() == str(user_id).strip()]
    if filtered.empty:
        return []

    records = filtered.sort_values("date", ascending=False).head(10).to_dict("records")
    for item in records:
        item["amount"] = int(float(item.get("amount", 0)))
    return records


def build_safe_context(user: Dict[str, Any]) -> str:
    account = get_safe_account(user)
    return (
        f"Name: {account['name']}\n"
        f"Account Type: {account['account_type']}\n"
        f"Account Number: {account['account_number']}\n"
        f"Balance: ₹{account['balance']:,}\n"
        f"Branch: {account['branch']}\n"
        f"IFSC: {account['ifsc']}\n"
        f"Account Status: {account['status']}"
    )
