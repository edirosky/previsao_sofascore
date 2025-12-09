#!/bin/bash
# @title run_once.sh - Executa todos os scripts uma vez

echo "=== Executando todos os scripts uma vez ==="
echo "Data/Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Ativa o ambiente virtual
if [ -d ".venv" ]; then
    echo "Ativando ambiente virtual..."
    source .venv/bin/activate
fi

# 2. Executa o scheduler uma vez
echo "Executando ciclo completo uma vez..."
python scripts/scheduler.py --cycles 1

# 3. Desativa ambiente virtual
if [ -d ".venv" ] && [ -n "$VIRTUAL_ENV" ]; then
    echo "Desativando ambiente virtual..."
    deactivate
fi

echo ""
echo "=== Execução concluída ==="
echo "Data/Hora: $(date '+%Y-%m-%d %H:%M:%S')"