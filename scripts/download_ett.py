import os
import requests

ETT_URLS = {
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
}


def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def main(name: str = "ETTm1"):
    if name not in ETT_URLS:
        raise ValueError(f"Unknown dataset {name}. Choose one of: {list(ETT_URLS.keys())}")

    out_path = f"data/{name}.csv"
    download(ETT_URLS[name], out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="ETTm1", choices=list(ETT_URLS.keys()))
    args = p.parse_args()
    main(args.name)
