import os
import json
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000/api"
OUT_DIR = Path("d:/stock-prediction/frontend/public/mock_api")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_save(url_path, filename):
    url = f"{BASE_URL}{url_path}"
    print(f"Fetching {url}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        out_path = OUT_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(resp.json(), f)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]

# Predict endpoints (defaults to AAPL)
fetch_and_save("/predict/AAPL", "predict/AAPL.json")
fetch_and_save("/models/AAPL", "models/AAPL.json")

# Stock endpoints (2y)
for t in tickers:
    fetch_and_save(f"/stock/{t}?period=2y", f"stock/{t}_2y.json")
    fetch_and_save(f"/returns/{t}?period=1y", f"returns/{t}_1y.json")

# Compare endpoint
fetch_and_save("/compare?tickers=AAPL,MSFT,NVDA,TSLA&period=1y", "compare_AAPL_MSFT_NVDA_TSLA_1y.json")
fetch_and_save("/compare?tickers=AAPL,MSFT&period=1y", "compare_AAPL_MSFT_1y.json")

print("Done generating static API data.")
