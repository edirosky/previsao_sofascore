import os
import ast
import sys
from collections import defaultdict

def extract_imports_from_py(file_path):
    imports = set()
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except Exception as e:
            print(f"Erro ao parsear {file_path}: {e}")
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    imports.add(module)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module:
                    imports.add(module.split('.')[0])
    return imports

def main():
    scripts_dir = "/workspaces/previsao_sofascore/scripts"
    all_imports = set()

    for root, _, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                pkgs = extract_imports_from_py(path)
                all_imports.update(pkgs)

    # Remover pacotes padrão do Python (opcional)
    stdlib = set(sys.stdlib_module_names)
    third_party = {pkg for pkg in all_imports if pkg not in stdlib and pkg != "__future__"}

    req_file = "/workspaces/previsao_sofascore/requirements.txt"
    with open(req_file, "w", encoding="utf-8") as f:
        for pkg in sorted(third_party):
            f.write(pkg + "\n")

    print(f"requirements.txt gerado com {len(third_party)} pacotes em {req_file}")

if __name__ == "__main__":
    main()
