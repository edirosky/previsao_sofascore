# @title CARREGAMENTO DE MODELOS - DF_CONCATENADOS COM CONFIANÇA AJUSTADA (VERSÃO GITHUB CODESPACE)

# ------------------ Instalar dependências ------------------
import subprocess
import sys

def instalar_pacotes(pacotes):
    for pacote in pacotes:
        try:
            __import__(pacote)
            print(f"✅ Pacote '{pacote}' já instalado.")
        except ImportError:
            print(f"⚡ Instalando '{pacote}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

pacotes_necessarios = [
    "pandas",
    "numpy",
    "joblib",
    "pytz",
    "IPython",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "xgboost"
]

instalar_pacotes(pacotes_necessarios)
print("\n🎉 Todas as dependências estão instaladas!")

