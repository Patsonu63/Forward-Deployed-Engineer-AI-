import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
# REST API CALLS — The backbone of all FDE integrations
# ══════════════════════════════════════════════════════

# ── 1. GET request — fetch data ───────────────────────
def get_github_user(username: str) -> dict:
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)

    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        return response.json()   # automatically parses JSON
    elif response.status_code == 404:
        print(f"User '{username}' not found")
        return {}
    else:
        print(f"Unexpected error: {response.status_code}")
        return {}

user = get_github_user("torvalds")
print(f"Name: {user.get('name')}")
print(f"Public repos: {user.get('public_repos')}")
print(f"Followers: {user.get('followers')}")


# ── 2. GET with query parameters ─────────────────────
def search_github_repos(query: str, per_page: int = 5) -> list:
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }
    # requests automatically appends params as ?q=...&sort=...
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        repos = data["items"]
        return [{"name": r["full_name"], "stars": r["stargazers_count"]} for r in repos]
    return []

repos = search_github_repos("langchain", per_page=3)
for r in repos:
    print(f"{r['name']} — {r['stars']:,} stars")


# ── 3. POST request — send data to an API ────────────
def create_mock_ticket(title: str, description: str, priority: str) -> dict:
    url = "https://jsonplaceholder.typicode.com/posts"   # free test API

    payload = {
        "title": title,
        "body": description,
        "userId": 1
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    # Note: use json=payload (not data=) — it auto-sets Content-Type

    print(f"POST status: {response.status_code}")  # 201 = Created
    return response.json()

ticket = create_mock_ticket(
    title="SSO Integration Failure",
    description="Enterprise client cannot authenticate via Okta SAML",
    priority="critical"
)
print(f"Created ticket ID: {ticket['id']}")


# ── 4. Authenticated request (Bearer token) ───────────
def call_authenticated_api(endpoint: str, token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.get(endpoint, headers=headers)
    return response.json() if response.status_code == 200 else {}


# ── 5. Robust API client with retries ─────────────────
import time

def robust_get(url: str, max_retries: int = 3, params: dict = None) -> dict | None:
    """
    Production-grade GET with retry logic.
    Handles rate limits (429) and server errors (5xx).
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                wait = 2 ** attempt          # exponential backoff: 1s, 2s, 4s
                print(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)

            elif response.status_code >= 500:
                print(f"Server error {response.status_code}. Retry {attempt+1}/{max_retries}")
                time.sleep(1)

            else:
                print(f"Client error: {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            print("Connection failed — check your internet")
        except requests.exceptions.Timeout:
            print(f"Request timed out (attempt {attempt+1})")

    print("All retries exhausted")
    return None

result = robust_get("https://api.github.com/users/anthropics")
if result:
    print(f"Anthropic GitHub: {result.get('name')} — {result.get('public_repos')} repos")
