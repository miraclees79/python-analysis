import os
import subprocess
from pathlib import Path

# --- KONFIGURACJA FILTRACJI ---
# Foldery, których AI nie powinno analizować (zmniejsza zużycie tokenów i szum)
IGNORE_DIRS = {
    '.git', '.github', '__pycache__', 'venv', '.venv', 'env', 
    'node_modules', 'build', 'dist', '.mypy_cache', '.pytest_cache', 
    'ai_context', 'tests', 'docs' 
}

# Rozszerzenia plików, które całkowicie ignorujemy (pliki binarne, dane, obrazy)
IGNORE_EXTENSIONS = {
    '.pyc', '.pyo', '.exe', '.dll', '.so', '.bin', '.dat', 
    '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz',
    '.json', '.csv', '.xlsx', '.sql', '.log'
}
# -----------------------------

def should_ignore(path: Path):
    """Sprawdza, czy ścieżka lub plik powinny zostać zignorowane."""
    # Sprawdź, czy jakikolwiek element ścieżki jest w liście IGNORE_DIRS
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    # Sprawdź rozszerzenie pliku
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    return False

def run_command(command, shell=True):
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e.stderr}")
        return f"Error executing {command}"

def main():
    output_dir = Path("ai_context")
    output_dir.mkdir(exist_ok=True)
    
    print("--- Generowanie przefiltrowanej struktury plików ---")
    # Zamiast prostego 'find', używamy rekurencji Pythona dla precyzyjnej filtracji
    structure_lines = []
    for path in Path('.').rglob('*'):
        if should_ignore(path):
            continue
        # Tworzymy wizualną reprezentację drzewa (uproszczoną)
        depth = len(path.relative_to('.').parts)
        indent = '  ' * (depth - 1)
        structure_lines.append(f"{indent}└── {path.name}")
    
    tree_structure = "\n".join(structure_lines)
    with open(output_dir / "project_structure.txt", "w", encoding="utf-8") as f:
        f.write(tree_structure)

    print("--- Generowanie Stub Files (.pyi) ---")
    run_command("pip install mypy")
    # Generujemy stuby w folderze tymczasowym, aby móc je przefiltrować
    run_command("stubgen -o ai_context/stubs .")

    print("--- Generowanie Mapy Zależności (Custom) ---")
    dependencies = []
    for path in Path('.').rglob('*.py'):
        if should_ignore(path):
            continue
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('from moj_system') or line.startswith('import moj_system'):
                    # Prosta ekstrakcja: plik aktualny -> co importuje
                    dependencies.append(f"{path.relative_to('.')} -> {line.strip()}")
    
    dep_graph = "\n".join(dependencies)
    with open(output_dir / "dependencies.txt", "w", encoding="utf-8") as f:
        f.write(dep_graph)

    print("--- Pakowanie przefiltrowanego kontekstu do MD ---")
    bundle_path = output_dir / "ai_context_bundle.md"
    with open(bundle_path, "w", encoding="utf-8") as bundle:
        bundle.write("# AI Context Bundle (Filtered)\n\n")
        
        bundle.write("## Project Structure\n")
        bundle.write("```text\n" + tree_structure + "\n```\n\n")
        
        bundle.write("## Interface Stubs (.pyi)\n")
        stubs_dir = output_dir / "stubs"
        for pyi_file in stubs_dir.rglob("*.pyi"):
            # Ważne: sprawdzamy, czy oryginalny plik nie był w zignorowanym folderze
            # stubgen tworzy strukturę odzwierciedlającą projekt
            original_path = Path('.') / pyi_file.relative_to(stubs_dir).with_suffix('.py')
            if should_ignore(original_path):
                continue
                
            bundle.write(f"### File: {pyi_file.relative_to(output_dir)}\n")
            bundle.write("```python\n")
            bundle.write(pyi_file.read_text(encoding="utf-8"))
            bundle.write("\n```\n\n")

        bundle.write("## Dependency Map (Custom Imports)\n")
        dep_file = output_dir / "dependencies.txt"
        if dep_file.exists():
            bundle.write("```text\n")
            bundle.write(dep_file.read_text(encoding="utf-8"))
            bundle.write("\n```\n\n")
        else:
            bundle.write("No dependencies found.\n\n")

    print(f"Sukces! Plik gotowy: {bundle_path}")

if __name__ == "__main__":
    main()