import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


SECONDS_PER_DAY = 24 * 60 * 60


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_datetime(value):
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def load_graph_rows(graph_csv_path):
    graph_csv_path = Path(graph_csv_path)
    with open(graph_csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        return list(reader)


def normalize_feature_rows(feature_rows):
    feature_tensor = torch.tensor(feature_rows, dtype=torch.float32)
    if feature_tensor.numel() == 0:
        return feature_tensor
    mean = feature_tensor.mean(dim=0, keepdim=True)
    std = feature_tensor.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (feature_tensor - mean) / std


def parse_graph_row(row):
    transaction_id = row.get("transaction_id", "")
    customer_id = row.get("customer_key", "")
    terminal_id = row.get("terminal_id") or row.get("authors_org") or "unknown_terminal"

    amount = safe_float(row.get("tx_amount"))
    time_seconds = safe_float(row.get("tx_time_seconds"))
    time_days = safe_float(row.get("tx_time_days"))
    tx_datetime = parse_datetime(row.get("tx_datetime"))

    if tx_datetime is not None:
        weekday_index = tx_datetime.weekday()
        seconds_of_day = tx_datetime.hour * 3600 + tx_datetime.minute * 60 + tx_datetime.second
    else:
        weekday_index = safe_int(time_days, 0) % 7
        seconds_of_day = time_seconds % SECONDS_PER_DAY

    weekday_one_hot = [0.0] * 7
    weekday_one_hot[weekday_index] = 1.0

    seconds_rad = 2.0 * math.pi * seconds_of_day / SECONDS_PER_DAY
    days_rad = 2.0 * math.pi * (time_days % 7.0) / 7.0
    features = [
        amount,
        math.log1p(max(amount, 0.0)),
        math.sin(seconds_rad),
        math.cos(seconds_rad),
        math.sin(days_rad),
        math.cos(days_rad),
    ] + weekday_one_hot

    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "terminal_id": str(terminal_id),
        "amount": amount,
        "seconds_of_day": seconds_of_day,
        "time_days": time_days,
        "weekday_index": weekday_index,
        "tx_datetime": tx_datetime,
        "sort_key": (time_days, seconds_of_day, transaction_id),
        "features": features,
    }


def load_parsed_transactions(graph_csv_path):
    rows = load_graph_rows(graph_csv_path)
    parsed_transactions = []
    seen_transaction_ids = set()
    for row in rows:
        parsed = parse_graph_row(row)
        if not parsed["transaction_id"] or not parsed["customer_id"]:
            continue
        if parsed["transaction_id"] in seen_transaction_ids:
            continue
        seen_transaction_ids.add(parsed["transaction_id"])
        parsed_transactions.append(parsed)
    parsed_transactions.sort(key=lambda item: item["sort_key"])
    return parsed_transactions


def filter_transactions(parsed_transactions, split_datetime=None, customer_ids=None, transaction_ids=None, include_before_split=True):
    customer_id_set = set(customer_ids) if customer_ids is not None else None
    transaction_id_set = set(transaction_ids) if transaction_ids is not None else None
    filtered_transactions = []
    for item in parsed_transactions:
        if split_datetime is not None and item["tx_datetime"] is not None:
            if include_before_split and item["tx_datetime"] >= split_datetime:
                continue
            if not include_before_split and item["tx_datetime"] < split_datetime:
                continue
        if customer_id_set is not None and item["customer_id"] not in customer_id_set:
            continue
        if transaction_id_set is not None and item["transaction_id"] not in transaction_id_set:
            continue
        filtered_transactions.append(item)
    return filtered_transactions


def aggregate_customer_features(parsed_transactions):
    if not parsed_transactions:
        return [0.0] * 16

    amounts = [item["amount"] for item in parsed_transactions]
    terminals = {item["terminal_id"] for item in parsed_transactions}
    weekday_sum = [0.0] * 7
    seconds_sin = 0.0
    seconds_cos = 0.0
    days_sin = 0.0
    days_cos = 0.0
    for item in parsed_transactions:
        weekday_sum[item["weekday_index"]] += 1.0
        seconds_rad = 2.0 * math.pi * item["seconds_of_day"] / SECONDS_PER_DAY
        days_rad = 2.0 * math.pi * (item["time_days"] % 7.0) / 7.0
        seconds_sin += math.sin(seconds_rad)
        seconds_cos += math.cos(seconds_rad)
        days_sin += math.sin(days_rad)
        days_cos += math.cos(days_rad)

    count = float(len(parsed_transactions))
    mean_amount = sum(amounts) / count
    variance = sum((amount - mean_amount) ** 2 for amount in amounts) / count
    return [
        count,
        mean_amount,
        math.sqrt(max(variance, 0.0)),
        float(len(terminals)),
        seconds_sin / count,
        seconds_cos / count,
        days_sin / count,
        days_cos / count,
        min(amounts),
    ] + [value / count for value in weekday_sum]


def aggregate_terminal_features(parsed_transactions):
    if not parsed_transactions:
        return [0.0] * 16

    amounts = [item["amount"] for item in parsed_transactions]
    customers = {item["customer_id"] for item in parsed_transactions}
    weekday_sum = [0.0] * 7
    seconds_sin = 0.0
    seconds_cos = 0.0
    days_sin = 0.0
    days_cos = 0.0
    for item in parsed_transactions:
        weekday_sum[item["weekday_index"]] += 1.0
        seconds_rad = 2.0 * math.pi * item["seconds_of_day"] / SECONDS_PER_DAY
        days_rad = 2.0 * math.pi * (item["time_days"] % 7.0) / 7.0
        seconds_sin += math.sin(seconds_rad)
        seconds_cos += math.cos(seconds_rad)
        days_sin += math.sin(days_rad)
        days_cos += math.cos(days_rad)

    count = float(len(parsed_transactions))
    mean_amount = sum(amounts) / count
    variance = sum((amount - mean_amount) ** 2 for amount in amounts) / count
    return [
        count,
        mean_amount,
        math.sqrt(max(variance, 0.0)),
        float(len(customers)),
        seconds_sin / count,
        seconds_cos / count,
        days_sin / count,
        days_cos / count,
        max(amounts),
    ] + [value / count for value in weekday_sum]


class FDHGraphCsvDataset:
    def __init__(self, graph_csv_path=None, customer_ids=None, parsed_transactions=None):
        self.graph_csv_path = Path(graph_csv_path) if graph_csv_path is not None else None
        if parsed_transactions is None:
            if self.graph_csv_path is None:
                raise ValueError("Either graph_csv_path or parsed_transactions must be provided.")
            parsed_transactions = load_parsed_transactions(self.graph_csv_path)
        self.rows_by_customer = defaultdict(list)
        for parsed in parsed_transactions:
            self.rows_by_customer[parsed["customer_id"]].append(parsed)

        if customer_ids is None:
            customer_ids = sorted(self.rows_by_customer.keys())
        self.customer_ids = [customer_id for customer_id in customer_ids if customer_id in self.rows_by_customer]
        self.graphs = [self._build_customer_graph(customer_id) for customer_id in self.customer_ids]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    def _build_customer_graph(self, customer_id):
        parsed_transactions = self.rows_by_customer[customer_id]
        transaction_ids = [item["transaction_id"] for item in parsed_transactions]
        terminal_ids = list(dict.fromkeys(item["terminal_id"] for item in parsed_transactions))
        terminal_index = {terminal_id: idx for idx, terminal_id in enumerate(terminal_ids)}

        customer_features = aggregate_customer_features(parsed_transactions)
        transaction_features = [item["features"] for item in parsed_transactions]
        terminal_to_transactions = defaultdict(list)
        for item in parsed_transactions:
            terminal_to_transactions[item["terminal_id"]].append(item)
        terminal_features = [
            aggregate_terminal_features(terminal_to_transactions[terminal_id])
            for terminal_id in terminal_ids
        ]

        customer_to_transaction_src = [0] * len(parsed_transactions)
        customer_to_transaction_dst = list(range(len(parsed_transactions)))
        transaction_to_terminal_src = list(range(len(parsed_transactions)))
        transaction_to_terminal_dst = [terminal_index[item["terminal_id"]] for item in parsed_transactions]

        data = HeteroData()
        data["customer"].x = torch.tensor([customer_features], dtype=torch.float32)
        data["transaction"].x = torch.tensor(transaction_features, dtype=torch.float32)
        data["terminal"].x = torch.tensor(terminal_features, dtype=torch.float32)

        customer_transaction_edge_index = torch.tensor(
            [customer_to_transaction_src, customer_to_transaction_dst], dtype=torch.long
        )
        transaction_terminal_edge_index = torch.tensor(
            [transaction_to_terminal_src, transaction_to_terminal_dst], dtype=torch.long
        )
        data[("customer", "owns", "transaction")].edge_index = customer_transaction_edge_index
        data[("transaction", "rev_owns", "customer")].edge_index = customer_transaction_edge_index.flip(0)
        data[("transaction", "at", "terminal")].edge_index = transaction_terminal_edge_index
        data[("terminal", "rev_at", "transaction")].edge_index = transaction_terminal_edge_index.flip(0)

        data.customer_id = customer_id
        data.transaction_ids = transaction_ids
        return data


class FDHGlobalGraphData:
    def __init__(self, graph_csv_path=None, parsed_transactions=None):
        self.graph_csv_path = Path(graph_csv_path) if graph_csv_path is not None else None
        if parsed_transactions is None:
            if self.graph_csv_path is None:
                raise ValueError("Either graph_csv_path or parsed_transactions must be provided.")
            parsed_transactions = load_parsed_transactions(self.graph_csv_path)

        customer_to_transactions = defaultdict(list)
        terminal_to_transactions = defaultdict(list)
        for parsed in parsed_transactions:
            customer_to_transactions[parsed["customer_id"]].append(parsed)
            terminal_to_transactions[parsed["terminal_id"]].append(parsed)

        self.parsed_transactions = list(parsed_transactions)
        self.customer_ids = sorted(customer_to_transactions.keys())
        self.transaction_ids = [item["transaction_id"] for item in self.parsed_transactions]
        self.transaction_customer_ids = [item["customer_id"] for item in self.parsed_transactions]
        self.terminal_ids = sorted(terminal_to_transactions.keys())

        customer_index = {customer_id: idx for idx, customer_id in enumerate(self.customer_ids)}
        terminal_index = {terminal_id: idx for idx, terminal_id in enumerate(self.terminal_ids)}

        customer_features = [
            aggregate_customer_features(customer_to_transactions[customer_id])
            for customer_id in self.customer_ids
        ]
        transaction_features = [item["features"] for item in self.parsed_transactions]
        terminal_features = [
            aggregate_terminal_features(terminal_to_transactions[terminal_id])
            for terminal_id in self.terminal_ids
        ]

        customer_to_transaction_src = []
        customer_to_transaction_dst = []
        transaction_to_terminal_src = []
        transaction_to_terminal_dst = []
        for transaction_idx, item in enumerate(self.parsed_transactions):
            customer_to_transaction_src.append(customer_index[item["customer_id"]])
            customer_to_transaction_dst.append(transaction_idx)
            transaction_to_terminal_src.append(transaction_idx)
            transaction_to_terminal_dst.append(terminal_index[item["terminal_id"]])

        data = HeteroData()
        data["customer"].x = normalize_feature_rows(customer_features)
        data["transaction"].x = normalize_feature_rows(transaction_features)
        data["terminal"].x = normalize_feature_rows(terminal_features)

        customer_transaction_edge_index = torch.tensor(
            [customer_to_transaction_src, customer_to_transaction_dst], dtype=torch.long
        )
        transaction_terminal_edge_index = torch.tensor(
            [transaction_to_terminal_src, transaction_to_terminal_dst], dtype=torch.long
        )
        data[("customer", "owns", "transaction")].edge_index = customer_transaction_edge_index
        data[("transaction", "rev_owns", "customer")].edge_index = customer_transaction_edge_index.flip(0)
        data[("transaction", "at", "terminal")].edge_index = transaction_terminal_edge_index
        data[("terminal", "rev_at", "transaction")].edge_index = transaction_terminal_edge_index.flip(0)

        data.customer_ids = self.customer_ids
        data.transaction_ids = self.transaction_ids
        data.transaction_customer_ids = self.transaction_customer_ids
        data.terminal_ids = self.terminal_ids
        self.data = data


def build_customer_graph_from_transactions(customer_id, parsed_transactions):
    dataset = FDHGraphCsvDataset(parsed_transactions=parsed_transactions, customer_ids=[customer_id])
    if len(dataset) == 0:
        return None
    return dataset[0]
