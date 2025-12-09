#!/bin/bash
# @title run_all.sh - Executa todos os scripts em sequência

echo "=== Iniciando execução de todos os scripts ==="
echo "Data/Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Ativa o ambiente virtual (se existir)
if [ -d ".venv" ]; then
    echo "Ativando ambiente virtual..."
    source .venv/bin/activate
fi

# 2. Executa o scheduler em modo contínuo com intervalo de 5 minutos
echo "Iniciando scheduler em modo contínuo..."
echo "Cada ciclo levará aproximadamente 2-3 minutos para processar"
echo "Intervalo entre ciclos: 5 minutos"
echo "Pressione Ctrl+C para parar"
echo ""

python scripts/scheduler.py --continuous --interval 5

# 3. Desativa ambiente virtual (se ativado)
if [ -d ".venv" ] && [ -n "$VIRTUAL_ENV" ]; then
    echo "Desativando ambiente virtual..."
    deactivate
fi

echo ""
echo "=== Execução concluída ==="
echo "Data/Hora: $(date '+%Y-%m-%d %H:%M:%S')"