#!/usr/bin/env python3
# @title scheduler.py - Executa scripts em sequência com delays
"""
Usage:
  python scripts/scheduler.py [--interval MINUTES] [--continuous]

Scheduler para executar scripts de coleta e processamento em sequência
com delays apropriados entre eles para garantir que os arquivos Excel
estejam disponíveis e atualizados.
"""

import subprocess
import time
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scheduler.log')
    ]
)

def run_script(script_name: str, description: str, timeout: int = 300) -> bool:
    """Executa um script Python e retorna True se sucesso."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logging.error(f"Script não encontrado: {script_name}")
        return False
    
    logging.info(f"=== Iniciando: {description} ({script_name}) ===")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logging.info(f"✓ {description} concluído em {elapsed:.1f}s")
            if result.stdout.strip():
                logging.debug(f"Saída:\n{result.stdout[:500]}...")
            return True
        else:
            logging.error(f"✗ {description} falhou (código {result.returncode})")
            if result.stderr:
                logging.error(f"Erro:\n{result.stderr[:500]}...")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"✗ {description} timeout após {timeout}s")
        return False
    except Exception as e:
        logging.error(f"✗ Erro ao executar {script_name}: {e}")
        return False

def run_cycle(cycle_num: int):
    """Executa um ciclo completo de scripts."""
    logging.info(f"\n{'='*60}")
    logging.info(f"CICLO #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*60}")
    
    # 1. Coletar dados dos jogos e baixar APIs
    success = run_script(
        "010_collect_and_process.py",
        "Coleta de eventos e download de APIs",
        timeout=600  # 10 minutos para coleta
    )
    
    if not success:
        logging.warning("Coleta falhou, pulando processamentos...")
        return False
    
    # Aguardar para garantir que arquivos Excel estejam disponíveis
    logging.info("Aguardando 30 segundos para estabilização de arquivos...")
    time.sleep(30)
    
    # 2. Reprocessar gols
    run_script(
        "011_reprocess_goals.py",
        "Reprocessamento de gols",
        timeout=300
    )
    
    # Delay entre processamentos
    logging.info("Aguardando 15 segundos...")
    time.sleep(5)
    
    # 3. Criar incidentes e estatísticas gerais
    run_script(
        "012_criar_incidentes_estatisticas_geral.py",
        "Criação de incidentes e estatísticas",
        timeout=300
    )
    
    # Delay entre processamentos
    logging.info("Aguardando 15 segundos...")
    time.sleep(5)
    
    # 4. Carregar modelos
    run_script(
        "020_carregar_modelos.py",
        "Carregamento de modelos",
        timeout=300
    )
    
    # 4. Carregar modelos
    run_script(
        "021_collect_and_process_depois_previsao.py",
        "Coleta e processamento depois da previsão",
        timeout=60
    )



    # Delay entre processamentos
    logging.info("Aguardando 15 segundos...")
    time.sleep(5)
    
    # 5. Reprocessar gols com registro
    run_script(
        "030_reprocessar_golos_com_registro_efetivo_refactor.py",
        "Reprocessamento de gols com registro",
        timeout=300
    )


    # 5. Reprocessar gols com registro
    run_script(
        "031_patch_simples_tipo_status.py",
        "Patch simples tipo status",
        timeout=300
    )




    # Delay entre processamentos
    logging.info("Aguardando 15 segundos...")
    time.sleep(5)
    
    # 6. Análise de métricas
    run_script(
        "040_analise_metricas_confianca.py",
        "Análise de métricas e confiança",
        timeout=300
    )

    # 6. Análise de métricas
    run_script(
        "041_dados_pontos_geral.py",
        "Análise de dados pontos geral",
        timeout=300
    )




    return True

def main():
    parser = argparse.ArgumentParser(description='Scheduler para scripts de análise SofaScore')
    parser.add_argument('--interval', type=int, default=2,
                       help='Intervalo entre ciclos em minutos (padrão: 2)')
    parser.add_argument('--continuous', action='store_true',
                       help='Executar continuamente em loop')
    parser.add_argument('--cycles', type=int, default=1,
                       help='Número de ciclos a executar (padrão: 1)')
    parser.add_argument('--start-delay', type=int, default=0,
                       help='Delay inicial em segundos (padrão: 0)')
    
    args = parser.parse_args()
    
    if args.start_delay > 0:
        logging.info(f"Aguardando {args.start_delay}s antes de iniciar...")
        time.sleep(args.start_delay)
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            success = run_cycle(cycle_count)
            
            # Verificar se deve continuar
            if not args.continuous and cycle_count >= args.cycles:
                logging.info(f"Concluído após {cycle_count} ciclo(s)")
                break
            
            # Calcular próximo ciclo
            next_run = datetime.now().timestamp() + (args.interval * 60)
            next_time = datetime.fromtimestamp(next_run).strftime('%H:%M:%S')
            
            logging.info(f"\nPróximo ciclo em {args.interval} minutos ({next_time})")
            logging.info("="*60 + "\n")
            
            # Aguardar intervalo, mas permitir interrupção
            for _ in range(args.interval * 60):
                time.sleep(1)
                
    except KeyboardInterrupt:
        logging.info("\nScheduler interrompido pelo usuário")
    except Exception as e:
        logging.error(f"Erro no scheduler: {e}")
    
    logging.info("Scheduler finalizado")

if __name__ == "__main__":
    main()