#!/bin/bash
cd /workspaces/previsao_sofascore
echo "🚀 Iniciando Frontend Streamlit..."
echo "📊 Verificando dados..."

# Verificar se os dados existem
if [ -f "data/df_previsoes_sim_concatenado.csv" ]; then
    echo "✅ Dados encontrados!"
    echo "📈 Total de linhas: $(wc -l < data/df_previsoes_sim_concatenado.csv)"
else
    echo "⚠️ Dados não encontrados. Execute primeiro:"
    echo "   python scripts/carregar_modelos.py"
    exit 1
fi

# Verificar dependências
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Instalando Streamlit..."
    pip install streamlit pandas numpy
fi

# Iniciar Streamlit
echo "🌐 Acesse: http://localhost:8501"
echo "🛑 Para parar: Ctrl+C"
echo "---"
streamlit run frontend/app/main.py --server.port=8501 --server.address=0.0.0.0