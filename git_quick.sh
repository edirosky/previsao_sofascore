#!/bin/bash
# Script rápido para commit e push

echo "🚀 Git Quick Commit & Push"
echo "=========================="

# Verificar status
echo -e "\n📊 Status atual:"
git status --short

# Adicionar tudo
echo -e "\n➕ Adicionando mudanças..."
git add .

# Criar mensagem com data
COMMIT_MSG="Atualizações - $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "\n📝 Mensagem de commit: $COMMIT_MSG"

# Fazer commit
git commit -m "$COMMIT_MSG"

# Obter branch atual
CURRENT_BRANCH=$(git branch --show-current)
echo -e "\n🌿 Branch atual: $CURRENT_BRANCH"

# Fazer push
echo -e "\n📤 Fazendo push..."
git push origin $CURRENT_BRANCH

echo -e "\n✅ Concluído!"
