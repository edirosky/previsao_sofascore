# @title 031_patch_simples_tipo_status.py - Atualizar status pós-previsão para jogos específicos
"""
Usage:
  python scripts/031_patch_simples_tipo_status.py

Atualiza apenas os jogos existentes no arquivo df_previsoes_sim_concatenado.csv
com informações de status coletadas via API.
"""

# @title Imports & Config
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from playwright.async_api import async_playwright, APIRequestContext
from camoufox.async_api import AsyncCamoufox

# ---------- Config ----------
ROOT = Path("./scripts/data")
ROOT.mkdir(parents=True, exist_ok=True)

PREVISOES_CSV = ROOT / 'df_previsoes_sim_concatenado.csv'
OUTPUT_CSV = ROOT / 'df_previsoes_sim_concatenado.csv'  # Mesmo arquivo (atualizar)

STORAGE_FILE = ROOT / "sofascore_cookies.json"
HEADLESS = True
LOG_LEVEL = logging.INFO

# logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Utilities ----------
def carregar_previsoes() -> pd.DataFrame:
    """Carrega o arquivo de previsões"""
    if not PREVISOES_CSV.exists():
        logging.error(f"Arquivo não encontrado: {PREVISOES_CSV}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(PREVISOES_CSV, dtype=str)
        logging.info(f"Carregado {PREVISOES_CSV} com {len(df)} linhas")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar {PREVISOES_CSV}: {e}")
        return pd.DataFrame()

def calcular_minutos_jogo(row: pd.Series) -> str:
    """Calcula minutos de jogo baseado no status e horários"""
    try:
        status = str(row.get("Status_depois_previsao", "")).lower()
        tipo_status = str(row.get("Tipo_Status_depois_previsao", "")).lower()
        
        # Status específicos
        if status in ["not started", "canceled", "postponed", "abandoned"]:
            return "00:00"
        
        if status == "halftime":
            return "45:00"
        
        if status in ["ended", "finished", "after penalties", "after extra time", "ft"]:
            return "90:00"
        
        # Se o jogo está em andamento
        inicio = row.get("Inicio_depois_previsao")
        atual_time = row.get("Atual_time_depois_previsao")
        
        if pd.notnull(atual_time) and str(atual_time).strip() != "":
            try:
                # Converter para datetime
                atual_dt = pd.to_datetime(atual_time, utc=True)
                agora = pd.to_datetime('now', utc=True)
                
                # Calcular diferença
                diff = agora - atual_dt
                
                # Adicionar 45 minutos para o segundo tempo
                if status.startswith("2nd") or tipo_status == "2ndhalf":
                    diff += timedelta(minutes=45)
                
                # Não permitir valores negativos
                if diff.total_seconds() < 0:
                    diff = timedelta(0)
                
                # Calcular minutos e segundos
                minutos = int(diff.total_seconds() // 60)
                segundos = int(diff.total_seconds() % 60)
                
                # Limitar a 90 minutos
                if minutos > 90:
                    minutos = 90
                    segundos = 0
                
                return f"{minutos:02d}:{segundos:02d}"
            except Exception as e:
                logging.debug(f"Erro ao calcular minutos para jogo {row.get('ID_Jogo')}: {e}")
                return "00:00"
        
        return "00:00"
    except Exception as e:
        logging.warning(f"Erro geral ao calcular minutos: {e}")
        return "00:00"

# ---------- Coletar dados de jogos específicos ----------
async def coletar_dados_jogos_especificos(ids_jogos: List[str]) -> Dict[str, Dict[str, Any]]:
    """Coleta dados apenas para os IDs de jogo específicos"""
    
    resultados = {}
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0",
        "accept-language": "en-US,en;q=0.9"
    }
    
    if not ids_jogos:
        logging.warning("Nenhum ID de jogo para coletar")
        return resultados
    
    logging.info(f"Coletando dados para {len(ids_jogos)} jogos específicos...")
    
    async with async_playwright() as pw:
        request_ctx: APIRequestContext = await pw.request.new_context()
        async with AsyncCamoufox(headless=HEADLESS) as browser:
            context = await (browser.new_context(storage_state=str(STORAGE_FILE), viewport={"width":1920,"height":1080}) 
                           if STORAGE_FILE.exists() else 
                           browser.new_context(viewport={"width":1920,"height":1080}))
            try:
                for game_id in ids_jogos:
                    try:
                        api_url = f"https://www.sofascore.com/api/v1/event/{game_id}"
                        api_resp = await request_ctx.get(api_url, headers=headers, timeout=30000)
                        
                        if api_resp.status == 200:
                            evento = await api_resp.json()
                            evento_data = evento.get('event', {})
                            
                            # Extrair apenas os campos necessários
                            status_desc = str(evento_data.get("status", {}).get("description", "")).lower()
                            status_type = str(evento_data.get("status", {}).get("type", "")).lower()
                            
                            # Inicio: startTimestamp
                            start_ts = evento_data.get('startTimestamp')
                            inicio = datetime.fromtimestamp(start_ts).isoformat() if start_ts else None
                            
                            # Atual_time: timestamp ou statusTime.timestamp
                            atual_ts = evento_data.get('timestamp')
                            if atual_ts is None and evento_data.get('statusTime', {}).get('timestamp'):
                                atual_ts = evento_data.get('statusTime', {}).get('timestamp')
                            
                            atual_time = datetime.fromtimestamp(atual_ts).isoformat() if atual_ts else None
                            
                            resultados[game_id] = {
                                'Status_depois_previsao': status_desc,
                                'Tipo_Status_depois_previsao': status_type,
                                'Inicio_depois_previsao': inicio,
                                'Atual_time_depois_previsao': atual_time
                            }
                            
                            logging.debug(f"Dados coletados para jogo {game_id}: {status_desc} ({status_type})")
                        else:
                            logging.warning(f"API retornou status {api_resp.status} para jogo {game_id}")
                            resultados[game_id] = None
                            
                    except Exception as e:
                        logging.warning(f"Erro ao coletar dados para jogo {game_id}: {e}")
                        resultados[game_id] = None
                
                # Salvar estado de armazenamento
                try:
                    tmp_ctx = await browser.new_context(viewport={"width":1920,"height":1080})
                    await tmp_ctx.storage_state(path=str(STORAGE_FILE))
                    await tmp_ctx.close()
                except Exception:
                    logging.debug("Falha ao salvar storage_state")
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
        await request_ctx.dispose()
    
    logging.info(f"Coleta concluída: {len([r for r in resultados.values() if r is not None])}/{len(ids_jogos)} jogos coletados")
    return resultados

# ---------- Atualizar previsões ----------
def atualizar_previsoes_com_dados(df_previsoes: pd.DataFrame, dados_jogos: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Atualiza o DataFrame de previsões com os dados coletados"""
    
    if df_previsoes.empty:
        logging.warning("DataFrame de previsões vazio")
        return df_previsoes
    
    # Garantir que ID_Jogo é string
    df_previsoes['ID_Jogo'] = df_previsoes['ID_Jogo'].astype(str)
    
    # Contadores
    atualizados = 0
    nao_encontrados = 0
    
    # Atualizar cada jogo
    for idx, row in df_previsoes.iterrows():
        game_id = str(row['ID_Jogo'])
        
        if game_id in dados_jogos and dados_jogos[game_id] is not None:
            dados = dados_jogos[game_id]
            
            # Atualizar colunas
            df_previsoes.at[idx, 'Status_depois_previsao'] = dados['Status_depois_previsao']
            df_previsoes.at[idx, 'Tipo_Status_depois_previsao'] = dados['Tipo_Status_depois_previsao']
            df_previsoes.at[idx, 'Inicio_depois_previsao'] = dados['Inicio_depois_previsao']
            df_previsoes.at[idx, 'Atual_time_depois_previsao'] = dados['Atual_time_depois_previsao']
            
            atualizados += 1
        else:
            nao_encontrados += 1
            logging.debug(f"Jogo {game_id} não encontrado ou sem dados")
    
    # Calcular minutos de jogo pós-previsão
    logging.info("Calculando minutos de jogo pós-previsão...")
    df_previsoes['Minutos_jogo_depois_previsao'] = df_previsoes.apply(calcular_minutos_jogo, axis=1)
    
    # Para jogos terminados, marcar como "Terminado"
    if 'Tipo_Status_depois_previsao' in df_previsoes.columns:
        mask_terminados = df_previsoes['Tipo_Status_depois_previsao'].apply(
            lambda x: str(x).lower() in ['ended', 'finished', 'ft'] if pd.notna(x) else False
        )
        df_previsoes.loc[mask_terminados, 'Minutos_jogo_depois_previsao'] = 'Terminado'
    
    logging.info(f"Atualização concluída: {atualizados} atualizados, {nao_encontrados} não encontrados")
    return df_previsoes

# ---------- Função principal ----------
async def main():
    """Função principal do script"""
    logging.info("Iniciando atualização de status pós-previsão...")
    
    # 1. Carregar previsões
    df_previsoes = carregar_previsoes()
    if df_previsoes.empty:
        logging.error("Não foi possível carregar o arquivo de previsões")
        return
    
    # 2. Extrair IDs de jogo únicos
    if 'ID_Jogo' not in df_previsoes.columns:
        logging.error("Coluna 'ID_Jogo' não encontrada no arquivo de previsões")
        return
    
    ids_jogos = df_previsoes['ID_Jogo'].dropna().unique().tolist()
    ids_jogos = [str(id_jogo).strip() for id_jogo in ids_jogos if str(id_jogo).strip()]
    
    if not ids_jogos:
        logging.warning("Nenhum ID de jogo encontrado no arquivo de previsões")
        return
    
    logging.info(f"Encontrados {len(ids_jogos)} IDs de jogo únicos")
    
    # 3. Coletar dados apenas para esses IDs
    dados_jogos = await coletar_dados_jogos_especificos(ids_jogos)
    
    if not dados_jogos:
        logging.warning("Nenhum dado coletado")
        return
    
    # 4. Atualizar DataFrame com dados coletados
    df_previsoes_atualizado = atualizar_previsoes_com_dados(df_previsoes, dados_jogos)
    
    # 5. Salvar arquivo atualizado
    df_previsoes_atualizado.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✅ Arquivo de previsões atualizado: {OUTPUT_CSV}")
    
    # 6. Mostrar estatísticas
    logging.info("\n=== ESTATÍSTICAS PÓS-ATUALIZAÇÃO ===")
    
    if 'Tipo_Status_depois_previsao' in df_previsoes_atualizado.columns:
        tipo_counts = df_previsoes_atualizado['Tipo_Status_depois_previsao'].value_counts()
        logging.info("\nTipo de Status pós-previsão:")
        for tipo, count in tipo_counts.items():
            if pd.notna(tipo):
                logging.info(f"  {tipo}: {count}")
    
    if 'Minutos_jogo_depois_previsao' in df_previsoes_atualizado.columns:
        minutos_counts = df_previsoes_atualizado['Minutos_jogo_depois_previsao'].value_counts().head(10)
        logging.info("\nMinutos de jogo pós-previsão (top 10):")
        for minutos, count in minutos_counts.items():
            if pd.notna(minutos):
                logging.info(f"  {minutos}: {count}")
    
    # 7. Mostrar amostra
    logging.info("\n=== AMOSTRA DOS DADOS ATUALIZADOS ===")
    colunas_mostrar = ['ID_Jogo', 'Time_Home', 'Time_Away', 'Tipo_Status_depois_previsao', 'Minutos_jogo_depois_previsao']
    colunas_disponiveis = [col for col in colunas_mostrar if col in df_previsoes_atualizado.columns]
    
    if colunas_disponiveis:
        print(df_previsoes_atualizado[colunas_disponiveis].head(10).to_string())
    
    logging.info(f"\n✅ Processamento concluído! Total: {len(df_previsoes_atualizado)} jogos")

# ---------- Execução ----------
if __name__ == "__main__":
    asyncio.run(main())