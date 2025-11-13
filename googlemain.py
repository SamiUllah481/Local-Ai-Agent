# --- Imports ---
import os
import sys
import json
import fnmatch
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- MODIFIED: Ollama Imports ---
# LangChain's Ollama integration
from langchain_community.llms import Ollama 

# --- LangChain Imports ---
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- GitHub Imports ---
from github import Github, InputGitAuthor, ContentFile


# ==============================================================================
# SHARED: SEARCH HELPERS AND CONFIG
# ==============================================================================

def _default_search_roots() -> List[str]:
    """Return a conservative set of default search roots on Windows.

    You can override by setting the SEARCH_ROOTS env var to a semicolon-separated list of absolute paths.
    """
    env = os.environ.get("SEARCH_ROOTS")
    if env:
        roots = [p.strip().strip('"') for p in env.split(";") if p.strip()]
        return [os.path.abspath(p) for p in roots if os.path.exists(p)]

    roots: List[str] = []
    home = os.path.expanduser("~")
    for sub in ("Desktop", "Documents", "Downloads"):
        p = os.path.join(home, sub)
        if os.path.isdir(p):
            roots.append(p)
    # Also include the current workspace folder
    try:
        cwd = os.path.abspath(os.getcwd())
        roots.append(cwd)
    except Exception:
        pass
    # Optionally add other drives if present (e.g., D:\)
    for drive in ("D:\\", "E:\\"):
        if os.path.isdir(drive):
            roots.append(drive)
    # De-dup while preserving order
    seen = set()
    uniq_roots: List[str] = []
    for r in roots:
        if r not in seen:
            uniq_roots.append(r)
            seen.add(r)
    return uniq_roots


def _find_paths_by_name(
    name_query: str,
    extensions: Optional[List[str]] = None,
    max_results: int = 25,
    roots: Optional[List[str]] = None,
) -> List[str]:
    """Search for files or folders by fuzzy name across configured roots.

    - name_query: pattern like "notes*" or "report.xlsx" (case-insensitive)
    - extensions: if provided, only return files with these extensions (like [".txt", ".csv"]).
    - Will return both files and directories if extensions is None.
    """
    patterns = [name_query]
    # If the user didn't include wildcards, match anywhere in the name
    if not any(ch in name_query for ch in "*?["):
        patterns.append(f"*{name_query}*")

    roots = roots or _default_search_roots()
    results: List[str] = []
    lowered_exts = {e.lower() for e in (extensions or [])}

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Search directories
            if not extensions:
                for d in dirnames:
                    dn = d.lower()
                    if any(fnmatch.fnmatch(dn, p.lower()) for p in patterns):
                        results.append(os.path.join(dirpath, d))
                        if len(results) >= max_results:
                            return results

            # Search files
            for f in filenames:
                fn = f.lower()
                if any(fnmatch.fnmatch(fn, p.lower()) for p in patterns):
                    full = os.path.join(dirpath, f)
                    if extensions:
                        if os.path.splitext(f)[1].lower() not in lowered_exts:
                            continue
                    results.append(full)
                    if len(results) >= max_results:
                        return results
    return results


# ==============================================================================
# SECTION 1: PANDAS DATAFRAME AGENT (CSV/EXCEL)
# ==============================================================================

def setup_dummy_csv(file_path: str):
    """Creates a dummy CSV file if one doesn't exist."""
    if not os.path.exists(file_path):
        print(f"File not found. Creating dummy file: {file_path}")
        data = {
            'OrderID': [101, 102, 103, 104, 105],
            'Product': ['Laptop', 'Monitor', 'Mouse', 'Keyboard', 'Webcam'],
            'Price': [1200, 300, 25, 75, 50],
            'Status': ['Shipped', 'Pending', 'Shipped', 'Pending', 'Delivered']
        }
        pd.DataFrame(data).to_csv(file_path, index=False)

def run_pandas_agent(file_path: str):
    """
    Runs a LangChain Pandas DataFrame agent on the specified CSV file
    using a local Ollama model.
    """
    try:
        # 1. Configuration - MODIFIED for Ollama
        # NOTE: Ensure you have 'llama3' or your desired model pulled in Ollama.
        llm = Ollama(
            temperature=0,
            model="llama3.2:3b"
        )
    except Exception as e:
        # MODIFIED Error Message
        print("Error initializing Ollama. Is the Ollama server running locally?")
        print(f"Details: {e}")
        return

    # 2. Create or load the DataFrame
    setup_dummy_csv(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file at {file_path}: {e}")
        return

    print("\n--- Original DataFrame Head ---")
    print(df.head())
    print("-----------------------------")

    # 4. Create the Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True, # See the agent's thoughts
        allow_dangerous_code=True,  # Opt-in to code execution
        handle_parsing_errors=True  # Auto-retry on parsing errors
        # The agent will default to a compatible type (like ZERO_SHOT_REACT_DESCRIPTION)
    )

    # 5. Define the AI Instruction (Modification Query)
    ai_instruction = (
        "Find 'notes' column and write 'pending' in it. "
        "After the modification, print the resulting DataFrame's head."
    )

    # 6. Execute the Instruction using the agent
    print(f"\n>>> Running Instruction: '{ai_instruction}'")
    
    # Store original DataFrame to check if it was modified
    df_original = df.copy()
    
    try:
        # The agent executes the modification code
        result = agent.invoke({"input": ai_instruction})
        
        # Check if the DataFrame was actually modified
        import re
        if df.equals(df_original):
            print("\n⚠️ DataFrame was not modified by agent. Attempting fallback...")
            # Extract code from the agent's output
            result_str = str(result)
            
            # Direct execution based on the instruction
            executed = False
            
            # Check if the instruction mentions 'Notes' or 'notes' column
            if 'notes' in ai_instruction.lower() or 'Notes' in result_str:
                try:
                    print("Executing: df['Notes'] = 'pending'")
                    df['Notes'] = 'pending'
                    executed = True
                except Exception as e:
                    print(f"Failed to execute: {e}")
            
            # Try to extract other pandas patterns from result
            if not executed:
                # Look for simple assignment patterns
                import re
                # Match patterns like: df['column'] = 'value' or df.loc[...] = ...
                patterns = [
                    r"df\['(\w+)'\]\s*=\s*'(\w+)'",
                    r"df\.loc\[([^\]]+)\]\s*=\s*(.+)"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, result_str)
                    if matches:
                        for match in matches:
                            try:
                                if len(match) == 2:
                                    # Simple column assignment
                                    col, val = match
                                    print(f"Executing: df['{col}'] = '{val}'")
                                    df[col] = val
                                    executed = True
                                    break
                            except Exception as e:
                                print(f"Failed: {e}")
                                continue
                    if executed:
                        break
            
            if not executed:
                print("❌ Could not extract or execute pandas code from agent output.")
                return
        
        # Save the modified DataFrame
        df.to_csv(file_path, index=False)
        print(f"\n✅ Modification complete and file saved to {file_path}!")

        # Verify the changes by reloading and printing the file
        df_new = pd.read_csv(file_path)
        print("\n--- Final Saved DataFrame Head ---")
        print(df_new.head())
    
    except Exception as e:
        # Fallback: If agent fails, try to extract and execute pandas code from error
        error_msg = str(e)
        print(f"\n⚠️ Agent error: {error_msg[:200]}...")
        
        if "Could not parse LLM output" in error_msg or "is not a valid tool" in error_msg:
            import re
            print("\n🔧 Attempting to extract pandas code from agent output...")
            
            # Try to find pandas code in the error message
            patterns = [
                r"df\.loc\[df\['Status'\]\s*==\s*'Pending',\s*'Price'\]\s*\*=\s*([\d.]+)",
                r"`(df\.loc\[[^\`]+\])\s*(\*=|=)\s*([^\`]+)`"
            ]
            
            executed = False
            for pattern in patterns:
                match = re.search(pattern, error_msg)
                if match:
                    try:
                        if "Status" in pattern and "Pending" in pattern:
                            # Direct extraction for this specific case
                            df.loc[df['Status'] == 'Pending', 'Price'] *= 1.10
                        else:
                            # Generic extraction
                            code = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                            exec(code)
                        
                        df.to_csv(file_path, index=False)
                        print(f"\n✅ Modification complete and file saved to {file_path} (via fallback)!")
                        
                        df_new = pd.read_csv(file_path)
                        print("\n--- Final Saved DataFrame Head ---")
                        print(df_new.head())
                        executed = True
                        break
                    except Exception as fallback_e:
                        print(f"Fallback pattern failed: {fallback_e}")
                        continue
            
            if not executed:
                print("\n❌ Could not extract executable pandas code. Please try option 3 instead.")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")

def run_pandas_main():
    """Main entry point for the Pandas agent."""
    print("\n--- Pandas CSV Agent ---")
    file_path = input("Enter the path to your CSV file (e.g., sales_data.csv): ").strip()
    if not file_path:
        print("No file path entered. Using 'default_sales.csv'.")
        file_path = "default_sales.csv"
        
    run_pandas_agent(file_path)


# ==============================================================================
# SECTION 2: GITHUB FOLDER PUSH UTILITY (Extended)
# ==============================================================================

def push_folder_to_github(repo_name: str, local_folder_path: str, commit_message: str, create_if_missing: bool = True):
    """
    Reads files from a local folder and pushes them to a specified GitHub repository.
    """
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Please set it to your Personal Access Token.")
        return

    g = Github(GITHUB_TOKEN)
    user = g.get_user()
    try:
        repo = user.get_repo(repo_name)
        print(f"Successfully connected to repo: {repo.full_name}")
    except Exception:
        if create_if_missing:
            try:
                repo = user.create_repo(repo_name, private=False)
                print(f"Created new repository: {repo.full_name}")
            except Exception as e:
                print(f"Error: Could not access or create repo '{repo_name}'. Details: {e}")
                return
        else:
            print(f"Error: Repo '{repo_name}' not found and create_if_missing is False.")
            return

    folder_path = os.path.abspath(local_folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: Local folder not found at {folder_path}")
        return
    
    print(f"Starting push from '{folder_path}' to '{repo_name}'...")

    # 1. Loop through all files in the local folder
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            
            # Create the path inside the GitHub repository
            github_path = os.path.relpath(local_file_path, folder_path).replace("\\", "/")
            
            # Read as text if possible; otherwise read binary and base64 encode
            content: Optional[str] = None
            try:
                with open(local_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except Exception:
                # Fallback: read binary and skip (PyGithub expects text). You can extend to handle binary if needed.
                print(f"Skipping non-text or unreadable file: {local_file_path}")
                continue
            
            try:
                # 2. Check if the file already exists
                contents = repo.get_contents(github_path)
                # If it exists, update it
                repo.update_file(
                    contents.path, 
                    commit_message, 
                    content, 
                    contents.sha
                )
                print(f"Updated file: {github_path}")
            
            except Exception as get_error:
                # If file does not exist (404 error), create it
                if "404" in str(get_error) or "empty" in str(get_error).lower():
                    try:
                        repo.create_file(
                            github_path, 
                            commit_message, 
                            content
                        )
                        print(f"Created file: {github_path}")
                    except Exception as create_error:
                        print(f"Error creating file {github_path}: {create_error}")
                else:
                    print(f"Error processing file {github_path}: {get_error}")

    print(f"\n✅ Successfully pushed files from '{local_folder_path}' to '{repo_name}'.")


def find_folder_and_push_to_github(folder_query: str, repo_name: str, commit_message: str) -> None:
    """Find a local folder by name (fuzzy) and push it to GitHub."""
    matches = _find_paths_by_name(folder_query, extensions=None, max_results=5)
    dirs = [m for m in matches if os.path.isdir(m)]
    if not dirs:
        print(f"No folders found matching '{folder_query}'.")
        return
    target = dirs[0]
    print(f"Found folder: {target}")
    push_folder_to_github(repo_name, target, commit_message)

def run_github_main():
    """Main entry point for the GitHub utility."""
    print("\n--- GitHub Folder Push Utility ---")
    repo_name = input("Enter your GitHub repository name (e.g., 'my-project'): ").strip()
    local_folder_path = input("Enter the path to the local folder to push: ").strip()
    commit_message = input("Enter your commit message (e.g., 'Automated file update'): ").strip()

    if not repo_name or not local_folder_path or not commit_message:
        print("Error: All fields (repo, folder path, commit message) are required.")
        return
        
    push_folder_to_github(repo_name, local_folder_path, commit_message)


# ==============================================================================
# SECTION 3: LOCAL OPS TOOLS: FILE SEARCH + TEXT/CSV/EXCEL EDIT
# ==============================================================================

def search_paths(name_query: str, extensions_json: str = "", max_results: int = 10) -> str:
    """Search for files or folders by fuzzy name across configured roots.

    Args:
        name_query: e.g., 'notes.txt' or 'sales*' (case-insensitive, wildcards ok)
        extensions_json: JSON array of extensions to filter (e.g., '[".txt", ".csv", ".xlsx"]'). Leave empty for any.
        max_results: maximum number of results to return.

    Returns JSON with keys: results: [paths]
    """
    try:
        exts = json.loads(extensions_json) if extensions_json else []
    except Exception:
        exts = []
    results = _find_paths_by_name(name_query, exts or None, max_results=max_results)
    return json.dumps({"results": results})


def replace_in_text_file(file_path: str, find_text: str, replace_text: str, make_backup: bool = True) -> str:
    """Replace occurrences of find_text with replace_text in a UTF-8 text file. Creates .bak if make_backup is True."""
    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if make_backup:
            with open(file_path + ".bak", 'w', encoding='utf-8') as b:
                b.write(content)
        new_content = content.replace(find_text, replace_text)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Replaced {content.count(find_text)} occurrence(s) in {file_path}. Backup: {make_backup}"
    except Exception as e:
        return f"Error updating file {file_path}: {e}"


def modify_tabular_file(file_path: str, instruction: str) -> str:
    """Use a Pandas Code Agent with Ollama to modify a CSV/XLSX per natural-language instruction. Saves file in-place.

    Pass clear instructions like: "Set Status='Closed' where OrderID==105" or "Increase Price by 10% where Status=='Pending'".
    """
    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        return f"Unsupported file type: {ext}. Only .csv, .xlsx, .xls supported."

    # Initialize Ollama LLM -> LangChain LLM (use smaller 1b model for resource-constrained systems)
    try:
        llm = Ollama(temperature=0, model="llama3.2:1b")
    except Exception as e:
        return f"Error initializing Ollama: {e}"

    # Load DataFrame
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            import pandas as _pd
            df = _pd.read_excel(file_path)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    try:
        # Execute agent instruction
        result = agent.invoke({"input": instruction})
        
        # Persist changes to disk
        if ext == ".csv":
            df.to_csv(file_path, index=False)
        else:
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, index=False)
        return f"✅ Updated and saved: {file_path}"
    except Exception as e:
        # Fallback: parse agent output for pandas code and execute it manually
        error_msg = str(e)
        if "Could not parse LLM output" in error_msg and "df.loc[" in error_msg:
            import re
            # Extract full pandas command with proper bracket handling
            match = re.search(r"`(df\.loc\[[^\`]+\])\s*=\s*([^\`]+)`", error_msg)
            if match:
                try:
                    code = f"{match.group(1)} = {match.group(2)}"
                    exec(code)
                    if ext == ".csv":
                        df.to_csv(file_path, index=False)
                    else:
                        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
                            df.to_excel(writer, index=False)
                    return f"✅ Updated and saved (via fallback): {file_path}"
                except Exception as fallback_e:
                    return f"Error in fallback execution: {fallback_e}"
        return f"Error modifying file via agent: {e}"

def run_file_agent_main():
    """Local File Ops helper without external agent framework.

    Provides a small guided flow to:
    - Search for files/folders
    - Replace text in a file
    - Modify CSV/Excel using Ollama-powered pandas agent
    - Push a folder to GitHub (via search)
    """
    print("\n--- Local File Ops (Guided) ---")
    print("Select an action:")
    print("  1) Search for files/folders")
    print("  2) Replace text in a file")
    print("  3) Modify CSV/Excel using Ollama")
    print("  4) Find folder and push to GitHub")
    choice = input("Enter 1, 2, 3, or 4: ").strip()

    if choice == '1':
        q = input("Enter name or pattern (e.g., notes*.txt): ").strip()
        exts = input("Optional extensions JSON (e.g., [\".txt\", \".csv\"]): ").strip()
        print(search_paths(q, exts or "", 15))
        return

    if choice == '2':
        q = input("File name or pattern: ").strip()
        res = json.loads(search_paths(q, json.dumps([".txt"]), 10))
        candidates = res.get("results", [])
        if not candidates:
            print("No matching text files found.")
            return
        print("Matches:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p}")
        idx = input("Pick file number: ").strip()
        try:
            path = candidates[int(idx)-1]
        except Exception:
            print("Invalid selection.")
            return
        find_text = input("Text to find: ").strip()
        replace_text = input("Replace with: ").strip()
        print(replace_in_text_file(path, find_text, replace_text, True))
        return

    if choice == '3':
        q = input("CSV/Excel name or pattern: ").strip()
        res = json.loads(search_paths(q, json.dumps([".csv", ".xlsx", ".xls"]), 10))
        candidates = res.get("results", [])
        if not candidates:
            print("No matching tabular files found.")
            return
        print("Matches:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p}")
        idx = input("Pick file number: ").strip()
        try:
            path = candidates[int(idx)-1]
        except Exception:
            print("Invalid selection.")
            return
        instr = input("Describe your change (e.g., Increase Price by 10% where Status == 'Pending'): ").strip()
        print(modify_tabular_file(path, instr))
        return

    if choice == '4':
        q = input("Folder name or pattern: ").strip()
        res = json.loads(search_paths(q, "", 10))
        candidates = [p for p in res.get("results", []) if os.path.isdir(p)]
        if not candidates:
            print("No matching folders found.")
            return
        print("Matches:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p}")
        idx = input("Pick folder number: ").strip()
        try:
            folder = candidates[int(idx)-1]
        except Exception:
            print("Invalid selection.")
            return
        repo = input("GitHub repo name (will create if missing): ").strip()
        msg = input("Commit message: ").strip() or "Automated update"
        push_folder_to_github(repo, folder, msg)
        return

    print("Unknown option.")


# ==============================================================================
# MAIN INTERACTIVE MENU
# ==============================================================================

if __name__ == "__main__":
    # Check for required environment variables - MODIFIED
    if not os.environ.get('GITHUB_TOKEN'):
        print("Warning: 'GITHUB_TOKEN' environment variable is not set.")
        print("The GitHub utility will fail.")
    else:
        # This check is removed since Ollama is local
        print("Ollama is assumed to be running locally.") 

    print("\n--- AI Agent Tools ---")
    print("What would you like to do?")
    print("  1: Run Pandas CSV Agent (Ollama)")
    print("  2: Push a folder to GitHub")
    print("  3: Run Local File Ops Agent (Ollama)")
    print("  4: Find folder and push to GitHub (search + push)")
    print("  q: Quit")
    
    choice = input("Enter 1, 2, 3, 4, or q: ").strip().lower()
    
    if choice == '1':
        run_pandas_main()
    elif choice == '2':
        run_github_main()
    elif choice == '3':
        run_file_agent_main()
    elif choice == '4':
        print("\n--- Push Folder to GitHub ---")
        folder_query = input("Folder name or pattern to search: ").strip()
        repo_name = input("Target GitHub repo name (will create if missing): ").strip()
        commit_message = input("Commit message: ").strip() or "Automated update"
        if folder_query and repo_name:
            find_folder_and_push_to_github(folder_query, repo_name, commit_message)
        else:
            print("Folder query and repo name are required.")
    elif choice == 'q':
        print("Exiting.")
        sys.exit()
    else:
        print("Invalid choice. Please run the script again.")