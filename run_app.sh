#!/bin/bash

# Navegar para o diretório do frontend
cd /workspaces/previsao_sofascore/frontend/app

# Verificar se o arquivo main.py existe
if [ ! -f "main.py" ]; then
    echo "❌ Arquivo main.py não encontrado!"
    exit 1
fi

# Instalar dependências se necessário
pip install -r ../requirements.txt 2>/dev/null || echo "⚠️  Verifique as dependências manualmente"

# Iniciar o Streamlit
echo "🚀 Iniciando Streamlit na porta 8501..."
echo "📊 Acesse: https://$(hostname)-8501.app.github.dev"
echo "📊 Ou: http://localhost:8501"

# Executar o Streamlit
streamlit run main.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.serverAddress="0.0.0.0" \
    --browser.serverPort=8501 \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --theme.textColor="#262730" \
    --theme.font="sans serif"
