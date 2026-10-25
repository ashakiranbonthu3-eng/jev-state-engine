#!/usr/bin/env python3
"""List all Gemini models available for the configured GOOGLE_API_KEY."""

import os
import sys
from pathlib import Path
from dotenv import dotenv_values

env_file = Path(".env")
env_vars = dotenv_values(env_file) if env_file.exists() else {}
api_key = os.environ.get("GOOGLE_API_KEY") or env_vars.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY is not set.")
    print("Set it in PowerShell with:")
    print('  $env:GOOGLE_API_KEY = "your_actual_key"')
    sys.exit(1)

try:
    from google import genai

    client = genai.Client(api_key=api_key)
    print("=" * 60)
    print("AVAILABLE GEMINI MODELS FOR YOUR API KEY:")
    print("=" * 60)
    found = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        if "gemini" in name.lower():
            found.append(name)
            print(f"  • {name}")

    print("=" * 60)
    if found:
        recommended = next((m for m in found if "2.0-flash" in m or "1.5-flash" in m or "flash" in m), found[0])
        print(f"Recommended model to use: --model {recommended}")
    else:
        print("No models found. Please verify your Google AI Studio project.")
    print("=" * 60)
except Exception as exc:
    print(f"Error querying Gemini models: {exc}")
    sys.exit(1)
