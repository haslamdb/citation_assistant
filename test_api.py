#!/usr/bin/env python3
"""Test API endpoints directly"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Login to get token
print("Logging in...")
response = requests.post(
    f"{BASE_URL}/api/token",
    data={"username": "admin", "password": "changeme"}
)

if response.status_code != 200:
    print(f"Login failed: {response.text}")
    exit(1)

token = response.json()["access_token"]
print(f"✓ Got token: {token[:20]}...")

# Headers with auth
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test search with multi_chunk
print("\nTesting search with multi_chunk...")
response = requests.post(
    f"{BASE_URL}/api/search",
    headers=headers,
    json={
        "query": "golgicide",
        "n_results": 2,
        "search_method": "multi_chunk"
    }
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✓ Got {data['n_results']} results")
else:
    print(f"✗ Error: {response.text}")

# Test summarize with factual
print("\nTesting summarize with factual search...")
response = requests.post(
    f"{BASE_URL}/api/summarize",
    headers=headers,
    json={
        "query": "how was golgicide discovered",
        "n_papers": 2,
        "search_method": "factual",
        "llm_model": "gemma2:27b"
    }
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✓ Got summary of length: {len(data.get('summary', ''))}")
else:
    print(f"✗ Error: {response.text[:500]}")