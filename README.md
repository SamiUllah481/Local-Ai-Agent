# Local AI Agent (Ollama)

This project provides a local AI agent powered by Ollama that can:

1. Find a text file on your PC and update its contents (find/replace)
2. Find and modify CSV/Excel files using pandas (via an agent that generates code)
3. Find a folder and push it to GitHub (creates the repo if missing)

All features are accessible via the interactive menu in `googlemain.py`.

## Prerequisites

- Windows, Python 3.10+ recommended
- [Ollama](https://ollama.com) installed and running locally
  - Pull a model, e.g.: `ollama pull llama3`
- GitHub Personal Access Token with `repo` scope set as environment variable `GITHUB_TOKEN`
- Git installed and available on PATH (only needed if you plan to use git locally; this script uses the GitHub API via PyGithub)

## Install

```powershell
# From this folder
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure search scope (optional but recommended)

Set `SEARCH_ROOTS` as a semicolon-separated list of folders to search, e.g.:

```powershell
$env:SEARCH_ROOTS = "C:\\Users\\<you>\\Documents;D:\\Projects"
```

If not set, the agent will search your Desktop, Documents, Downloads, the current workspace folder, and available D:/E: drives.

## Run

```powershell
python googlemain.py
```

Menu options:
- 1: Run Pandas CSV Agent (example)
- 2: Push a folder to GitHub (manual path)
- 3: Run Local File Ops Agent (natural language + tools)
- 4: Find folder by name and push to GitHub (search + push)

## Examples

- Text file replace (option 3):
  - "Replace 'foo' with 'bar' in notes.txt"
- CSV/Excel modify (option 3):
  - "Increase Price by 10% where Status == 'Pending' in sales.xlsx"
- Push folder to GitHub (option 4):
  - Folder name: `my-app`  → Repo name: `my-app` → Commit: `Initial import`

## Notes & Safety

- Text edits create a `.bak` backup by default.
- Binary files are skipped during GitHub push via API (only text handled).
- The CSV/Excel agent executes generated pandas operations on a temporary DataFrame and writes back to the file.
- For Excel, only the first sheet is written back in this simple flow.

## Troubleshooting

- Ensure Ollama is running: open a terminal and run `ollama list` to verify.
- The CSV/Excel agent uses `llama3.2:1b` by default (small, fast). For better accuracy, edit `googlemain.py` to use a larger model like `llama3.1:8b` if you have sufficient memory.
- If the CSV/Excel tool says unsupported extension, ensure the file ends with `.csv` or `.xlsx`/`.xls`.
- For GitHub: set `GITHUB_TOKEN` in the current shell, e.g.:

```powershell
$env:GITHUB_TOKEN = "ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

- If search is slow, restrict `SEARCH_ROOTS` to a smaller set of directories.

## Known Issues

- **CSV Agent with llama3.2:1b**: The 1b model sometimes produces malformed output that can't be parsed. The code includes a fallback regex extractor, but it's not 100% reliable. Use a larger model (3b/8b) for production.
- **Python 3.14**: Pydantic v1 compatibility warnings appear but don't affect functionality.
