import csv
import json
import os
import random

import numpy as np
import torch

def flatten_records_for_csv(records):
    flat_rows = []
    for record in records:
        row = dict(record)
        if isinstance(row.get("datasets"), list):
            row["datasets"] = "|".join(row["datasets"])
        flat_rows.append(row)
    return flat_rows


def write_records(report_dir, file_stem, records):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, file_stem + ".json")
    csv_path = os.path.join(report_dir, file_stem + ".csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    rows = flatten_records_for_csv(records)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)


def log_scalars(writer, prefix, scalar_map, epoch):
    if writer is None:
        return
    for key, value in scalar_map.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
