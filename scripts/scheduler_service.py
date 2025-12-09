#!/usr/bin/env python3
# @title scheduler_service.py - Serviço de scheduler com monitoramento
"""
Serviço mais robusto com monitoramento de memória e restart automático.
"""

import subprocess
import time
import logging
import psutil
import signal
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SERVICE] %(message)s"
)

class ScriptRunner:
    def __init__(self, script_path, max_memory_mb=500):
        self.script_path = script_path
        self.max_memory_mb = max_memory_mb
        self.process = None
        
    def start(self):
        """Inicia o script."""
        self.process = subprocess.Popen(
            [sys.executable, self.script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Script iniciado: {self.script_path} (PID: {self.process.pid})")
        
    def stop(self):
        """Para o script."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logging.info(f"Script terminado: {self.script_path}")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logging.warning(f"Script forçado a parar: {self.script_path}")
                
    def check_memory(self):
        """Verifica uso de memória."""
        if not self.process:
            return True
            
        try:
            proc = psutil.Process(self.process.pid)
            memory_mb = proc.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                logging.warning(f"Memória alta: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                return False
            return True
        except psutil.NoSuchProcess:
            return True

class SchedulerService:
    def __init__(self, interval_minutes=5):
        self.interval = interval_minutes * 60
        self.running = True
        self.scripts = []
        
        # Configurar handler para SIGTERM/SIGINT
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handler para sinais de término."""
        logging.info(f"Recebido sinal {signum}, parando serviço...")
        self.running = False
        
    def add_script(self, script_name, description, timeout=300):
        """Adiciona script à fila."""
        self.scripts.append({
            'name': script_name,
            'description': description,
            'timeout': timeout
        })
        
    def run(self):
        """Executa o serviço principal."""
        logging.info("=== Iniciando Scheduler Service ===")
        
        # Configurar scripts
        self.add_script("010_collect_and_process.py", "Coleta de dados", 100)
        self.add_script("011_reprocess_goals.py", "Reprocessamento de gols", 100)
        self.add_script("012_criar_incidentes_estatisticas_geral.py", "Estatísticas", 100)
        self.add_script("020_carregar_modelos.py", "Carregar modelos", 100)
        self.add_script("021_collect_and_process_depois_previsao.py", "Carregar modelos", 100)        
        self.add_script("030_reprocessar_golos_com_registro_efetivo_refactor.py", "Gols com registro", 100)
        self.add_script("031_patch_simples_tipo_status.py", "Gols com registro", 100)
        self.add_script("040_analise_metricas_confianca.py", "Análise métricas", 100)
        self.add_script("041_dados_pontos_geral", "Análise métricas", 100)


        
        cycle_count = 0
        
        while self.running:
            cycle_count += 1
            cycle_start = datetime.now()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"CICLO #{cycle_count} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"{'='*60}")
            
            # Executar cada script
            for script_info in self.scripts:
                if not self.running:
                    break
                    
                script_name = script_info['name']
                description = script_info['description']
                timeout = script_info['timeout']
                
                logging.info(f"Executando: {description}")
                
                runner = ScriptRunner(f"scripts/{script_name}")
                runner.start()
                
                # Aguardar conclusão com timeout
                start_time = time.time()
                while runner.process.poll() is None:
                    if time.time() - start_time > timeout:
                        logging.error(f"Timeout: {script_name}")
                        runner.stop()
                        break
                        
                    # Verificar memória
                    if not runner.check_memory():
                        logging.error(f"Memória excessiva: {script_name}")
                        runner.stop()
                        break
                        
                    time.sleep(1)
                
                # Aguardar entre scripts
                if self.running:
                    logging.info(f"Aguardando 15s...")
                    time.sleep(5)
            
            # Calcular tempo até próximo ciclo
            if self.running:
                elapsed = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(0, self.interval - elapsed)
                
                if wait_time > 0:
                    next_time = (datetime.now().timestamp() + wait_time)
                    next_str = datetime.fromtimestamp(next_time).strftime('%H:%M:%S')
                    
                    logging.info(f"Próximo ciclo em {wait_time:.0f}s ({next_str})")
                    
                    # Aguardar com verificação periódica
                    for _ in range(int(wait_time)):
                        if not self.running:
                            break
                        time.sleep(1)
        
        logging.info("=== Scheduler Service finalizado ===")

def main():
    service = SchedulerService(interval_minutes=5)
    service.run()

if __name__ == "__main__":
    main()