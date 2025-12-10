# @title Auto-Scan + Installer de Dependências (para Codespaces)
"""
O que faz:
1. Varre recursivamente o repositório à procura de arquivos .py
2. Analisa imports (ast) e cria uma lista de módulos usados
3. Tradução heurística para nomes PyPI (mapa manual para casos comuns)
4. Verifica pacotes instalados e instala os que faltam
5. Atualiza requirements.txt com pip freeze

Como usar:
- Coloca este ficheiro em scripts/setup_environment_autoscan.py
- Corre: python scripts/setup_environment_autoscan.py
"""

import ast
import os
import subprocess
import sys
import pkgutil
from pathlib import Path
from typing import Set, Dict

# --- CONFIGURAÇÃO ---
ROOT_DIR = Path(".")  # altera se precisaes apenas de uma subpasta
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules"}
REQUIREMENTS_FILE = Path("requirements.txt")
# Mapa manual: nomes de módulo -> pacote PyPI (adiciona conforme necessário)
PYPI_MAP: Dict[str, str] = {
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "asyncio": "asyncio",  # built-in, mas deixo exemplo
    # adiciona outras traduções que uses frequentemente...
}

# Módulos que sabemos serem da stdlib e não devemos incluir
# (lista curta — podemos confiar também em pkgutil para detectar instalados)
STDLIB_EXCEPTIONS = {
    "sys", "os", "re", "json", "math", "time", "datetime", "pathlib", "logging",
    "asyncio", "collections", "itertools", "functools", "typing", "subprocess",
    "ast", "inspect", "threading", "http", "unittest", "xml", "csv"
}


def find_py_files(root: Path) -> Set[Path]:
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # filtrar dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if fname.endswith(".py"):
                files.add(Path(dirpath) / fname)
    return files


def extract_imports_from_file(py_path: Path) -> Set[str]:
    text = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except Exception:
        return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root_name = n.name.split(".")[0]
                imports.add(root_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_name = node.module.split(".")[0]
                imports.add(root_name)
    return imports


def aggregate_imports(py_files: Set[Path]) -> Set[str]:
    all_imps = set()
    for f in py_files:
        imps = extract_imports_from_file(f)
        all_imps.update(imps)
    return all_imps


def map_to_pypi(modules: Set[str]) -> Set[str]:
    pkgs = set()
    for m in modules:
        if m in STDLIB_EXCEPTIONS:
            continue
        if m.startswith("_"):
            continue
        # Se estiver no mapa manual, usa-o
        if m in PYPI_MAP:
            pkgs.add(PYPI_MAP[m])
        else:
            # heurística: lowercase do nome do módulo
            pkgs.add(m.lower())
    return pkgs


def get_installed_packages() -> Set[str]:
    installed = {m.name.lower() for m in pkgutil.iter_modules()}
    return installed


def pip_install(packages):
    if not packages:
        return
    python = sys.executable
    for pkg in sorted(packages):
        try:
            print(f"📦 Instalando {pkg} ...")
            subprocess.check_call([python, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Falhou ao instalar {pkg}: {e}. Continuando...")


def write_requirements():
    python = sys.executable
    with open(REQUIREMENTS_FILE, "w", encoding="utf-8") as fh:
        subprocess.check_call([python, "-m", "pip", "freeze"], stdout=fh)
    print(f"✅ {REQUIREMENTS_FILE} atualizado.")


def main():
    print("🔎 Escaneando arquivos .py ...")
    py_files = find_py_files(ROOT_DIR)
    print(f"→ Encontrados {len(py_files)} ficheiros Python para analisar.")
    modules = aggregate_imports(py_files)
    print(f"→ Módulos detectados (exemplo): {sorted(list(modules))[:20]}")
    candidate_pkgs = map_to_pypi(modules)
    print(f"→ Possíveis pacotes PyPI: {sorted(list(candidate_pkgs))[:40]}")

    installed = get_installed_packages()
    to_install = [p for p in candidate_pkgs if p.lower() not in installed]

    if not to_install:
        print("✅ Nenhum pacote faltante detectado (baseado em pkgutil).")
    else:
        print(f"🔁 Instalando {len(to_install)} pacotes faltantes...")
        pip_install(to_install)

    # Regenerar requirements.txt
    print("📝 A gerar/atualizar requirements.txt ...")
    write_requirements()
    print("🎉 Concluído.")


if __name__ == "__main__":
    main()
