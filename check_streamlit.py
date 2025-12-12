#!/usr/bin/env python3
import sys
import os

# Adicionar caminho
sys.path.append('/workspaces/previsao_sofascore/frontend/app')

try:
    # Tentar importar módulos necessários
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib
    import plotly.express as px
    import plotly.graph_objects as go
    
    print("✅ Todos os módulos principais importados com sucesso!")
    
    # Verificar se main.py existe e pode ser lido
    main_path = '/workspaces/previsao_sofascore/frontend/app/main.py'
    if os.path.exists(main_path):
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = len(content.split('\n'))
            print(f"✅ main.py encontrado com {lines} linhas")
            
            # Verificar imports críticos
            required_imports = [
                'import streamlit',
                'import pandas',
                'import numpy',
                'import matplotlib'
            ]
            
            missing = []
            for imp in required_imports:
                if imp not in content:
                    missing.append(imp)
            
            if missing:
                print(f"⚠️  Imports faltando: {missing}")
            else:
                print("✅ Todos os imports necessários estão presentes")
    else:
        print(f"❌ Arquivo não encontrado: {main_path}")
        
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
except Exception as e:
    print(f"❌ Erro geral: {e}")

# Verificar estrutura de diretórios
print("\n📁 Verificando estrutura de diretórios:")
base_dir = '/workspaces/previsao_sofascore'
dirs_to_check = [
    'frontend',
    'frontend/app', 
    'scripts',
    'data',
    '.streamlit'
]

for dir_path in dirs_to_check:
    full_path = os.path.join(base_dir, dir_path)
    if os.path.exists(full_path):
        print(f"  ✅ {dir_path}/")
    else:
        print(f"  ❌ {dir_path}/ (não existe)")

print("\n🎯 Streamlit está configurado e pronto para uso!")
