#!/usr/bin/env python3
# @title wait_for_csv.py - Espera por arquivos CSV

import os
import glob
import time
import sys

def wait_for_csv_files(min_files=1, data_dir="data", timeout=60):
    """
    Espera até que existam arquivos CSV no diretório.
    
    Args:
        min_files: Número mínimo de arquivos CSV esperados
        data_dir: Diretório onde procurar
        timeout: Tempo máximo de espera em segundos
    """
    print(f"⏳ Esperando por pelo menos {min_files} arquivo(s) CSV em {data_dir}...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # Filtrar arquivos não vazios
        valid_csv_files = [f for f in csv_files if os.path.getsize(f) > 0]
        
        if len(valid_csv_files) >= min_files:
            elapsed = time.time() - start_time
            print(f"✅ Encontrados {len(valid_csv_files)} arquivos CSV após {elapsed:.1f}s:")
            for f in valid_csv_files:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                print(f"  • {os.path.basename(f)} ({size_mb:.2f} MB)")
            return True
        
        # Mostrar progresso a cada 5 segundos
        if int(time.time() - start_time) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Aguardando... ({elapsed:.0f}/{timeout}s)")
        
        time.sleep(1)
    
    print(f"⏰ Timeout: Nenhum arquivo CSV encontrado após {timeout}s")
    return False

if __name__ == "__main__":
    # Usar argumentos da linha de comando se fornecidos
    min_files = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    success = wait_for_csv_files(min_files=min_files, timeout=timeout)
    sys.exit(0 if success else 1)