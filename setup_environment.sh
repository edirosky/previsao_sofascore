#!/bin/bash
echo "Configurando ambiente Python..."

# Verificar Python
python --version

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python -m venv venv
fi

# Ativar ambiente virtual
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Instalar dependências básicas
echo "Instalando dependências..."
pip install --upgrade pip
pip install matplotlib streamlit pandas numpy scikit-learn

# Se existir requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
fi

echo "Ambiente configurado com sucesso!"
echo "Para ativar manualmente: source venv/bin/activate"