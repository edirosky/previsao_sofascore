#!/usr/bin/env python3
"""
Script para automatizar o processo de Git:
- Verificar status
- Adicionar mudanças
- Fazer commit com mensagem dinâmica
- Verificar branch
- Fazer push
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description=None):
    """Executa um comando no terminal e imprime resultado"""
    if description:
        print(f"\n📋 {description}")
        print(f"💻 Comando: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Saída:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  Erros:\n{result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar: {command}")
        print(f"📄 Saída de erro:\n{e.stderr}")
        return None

def get_git_status():
    """Obtém o status atual do Git"""
    status = run_command("git status", "Verificando status do Git")
    return status

def get_current_branch():
    """Obtém o branch atual"""
    branch_output = run_command("git branch --show-current", "Obtendo branch atual")
    if branch_output:
        return branch_output.strip()
    return None

def create_commit_message(base_msg):
    """Cria uma mensagem de commit com data/hora"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{base_msg} - {now}"

def main():
    """Função principal"""
    print("🚀 INICIANDO AUTOMAÇÃO GIT")
    print("=" * 50)
    
    # 1. Verificar se estamos em um repositório Git
    if not os.path.exists(".git"):
        print("❌ Diretório atual não é um repositório Git!")
        sys.exit(1)
    
    # 2. Obter status
    status = get_git_status()
    if not status:
        print("❌ Não foi possível obter status do Git")
        sys.exit(1)
    
    # 3. Perguntar se quer continuar
    print("\n🔍 Status atual:")
    run_command("git status --short")
    
    confirm = input("\n❓ Deseja continuar com o commit? (s/N): ").strip().lower()
    if confirm != 's':
        print("❌ Operação cancelada pelo usuário")
        sys.exit(0)
    
    # 4. Adicionar todas as mudanças
    print("\n" + "=" * 50)
    run_command("git add .", "Adicionando todas as mudanças")
    
    # 5. Criar mensagem de commit
    print("\n" + "=" * 50)
    default_msg = "Atualizações do frontend: layout horizontal, scheduler, configuração JSON"
    
    custom_msg = input(f"\n📝 Mensagem de commit [padrão: '{default_msg}']: ").strip()
    if not custom_msg:
        custom_msg = default_msg
    
    commit_msg = create_commit_message(custom_msg)
    
    # 6. Fazer commit
    run_command(f'git commit -m "{commit_msg}"', f"Fazendo commit: {commit_msg}")
    
    # 7. Verificar branch atual
    print("\n" + "=" * 50)
    current_branch = get_current_branch()
    if current_branch:
        print(f"🌿 Branch atual: {current_branch}")
    else:
        current_branch = "main"  # Assume main como padrão
    
    # 8. Perguntar se quer fazer push
    print("\n" + "=" * 50)
    push_confirm = input(f"❓ Deseja fazer push para o branch '{current_branch}'? (s/N): ").strip().lower()
    
    if push_confirm == 's':
        run_command(f"git push origin {current_branch}", f"Fazendo push para origin/{current_branch}")
    
    # 9. Resumo final
    print("\n" + "=" * 50)
    print("✅ PROCESSO CONCLUÍDO!")
    print(f"📝 Commit: {commit_msg}")
    print(f"🌿 Branch: {current_branch}")
    print("📤 Push: " + ("✅ Concluído" if push_confirm == 's' else "⏸️  Pendente"))
    
    # 10. Mostrar status final
    print("\n📊 Status final:")
    run_command("git status --short")

if __name__ == "__main__":
    main()
