#!/usr/bin/env python3
"""
Trakt List ID Lookup
====================
Extracts the numeric Trakt list ID from any Trakt list URL.

Usage:
    python trakt_list_id.py <trakt_list_url> [--client-id YOUR_CLIENT_ID]

Examples:
    python trakt_list_id.py https://trakt.tv/users/justin/lists/imdb-top-rated-movies
    python trakt_list_id.py https://trakt.tv/users/justin/lists/imdb-top-rated-movies --client-id abc123

You can also set the TRAKT_CLIENT_ID environment variable instead of passing --client-id.
To get a client ID, register a free app at https://trakt.tv/oauth/applications
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error


def parse_trakt_url(url: str) -> tuple[str, str]:
    """Parse a Trakt list URL and return (username, list_slug)."""
    # Handle both full URLs and partial paths
    # Patterns:
    #   https://trakt.tv/users/{user}/lists/{slug}
    #   https://app.trakt.tv/users/{user}/lists/{slug}
    #   https://trakt.tv/users/{user}/lists/{slug}/items
    #   /users/{user}/lists/{slug}
    pattern = r"(?:https?://(?:[a-z]+\.)?trakt\.tv)?/users/([^/]+)/lists/([^/?\s#]+)"
    match = re.search(pattern, url.strip())

    if not match:
        print(f"Error: Could not parse Trakt list URL: {url}")
        print("Expected format: https://trakt.tv/users/<username>/lists/<list-slug>")
        sys.exit(1)

    return match.group(1), match.group(2)


def get_list_info(username: str, list_slug: str, client_id: str) -> dict:
    """Fetch list details from the Trakt API."""
    api_url = f"https://api.trakt.tv/users/{username}/lists/{list_slug}"

    req = urllib.request.Request(api_url)
    req.add_header("Content-Type", "application/json")
    req.add_header("trakt-api-version", "2")
    req.add_header("trakt-api-key", client_id)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        if e.code == 404:
            print(f"Error: List not found — users/{username}/lists/{list_slug}")
        elif e.code == 401:
            print("Error: Invalid or missing Trakt Client ID.")
            print("Get one free at https://trakt.tv/oauth/applications")
        elif e.code == 403:
            print(f"Error: Access forbidden — users/{username}/lists/{list_slug}")
            print("Possible causes:")
            print("  • The list is private")
            print("  • The user's profile is private")
            print("  • The Client ID is invalid or the app is not approved")
        else:
            print(f"Error: Trakt API returned HTTP {e.code}")
        if body:
            print(f"API response: {body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: Could not reach Trakt API — {e.reason}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Get the numeric Trakt list ID from a list URL."
    )
    parser.add_argument("url", help="Trakt list URL")
    parser.add_argument(
        "--client-id",
        default=os.environ.get("TRAKT_CLIENT_ID"),
        help="Trakt API Client ID (or set TRAKT_CLIENT_ID env var)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output full list info as JSON",
    )
    args = parser.parse_args()

    if not args.client_id:
        print("Error: No Trakt Client ID provided.")
        print("  Pass --client-id <ID> or set the TRAKT_CLIENT_ID environment variable.")
        print("  Register a free app at https://trakt.tv/oauth/applications")
        sys.exit(1)

    username, list_slug = parse_trakt_url(args.url)
    print(f"  Parsed:     users/{username}/lists/{list_slug}")
    print()
    data = get_list_info(username, list_slug, args.client_id)

    if args.output_json:
        print(json.dumps(data, indent=2))
    else:
        trakt_id = data.get("ids", {}).get("trakt")
        slug = data.get("ids", {}).get("slug")
        name = data.get("name", "Unknown")
        item_count = data.get("item_count", "?")
        user = data.get("user", {}).get("username", username)

        print(f"  Name:       {name}")
        print(f"  User:       {user}")
        print(f"  Slug:       {slug}")
        print(f"  Trakt ID:   {trakt_id}")
        print(f"  Items:      {item_count}")


if __name__ == "__main__":
    main()