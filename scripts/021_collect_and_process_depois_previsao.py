# @title collect_and_process.py - Coletar eventos -> Baixar APIs JSON -> Processar estatísticas (1ST half)
"""
Usage:
  python scripts/collect_and_process.py

Requirements:
  - playwright
  - camoufox
  - aiohttp
  - pandas
  - openpyxl
  - tqdm
"""

# @title Imports & Config
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib
import concurrent.futures

import aiohttp
import pandas as pd

from playwright.async_api import async_playwright, BrowserContext, APIRequestContext
from camoufox.async_api import AsyncCamoufox

# ---------- Config ----------
ROOT = Path('./data')
ROOT.mkdir(parents=True, exist_ok=True)

BASE = ROOT / 'playwright_jsons_full'
BASE.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = BASE / 'registry_estatisticas_processed.json'
JOGOS_XLSX = ROOT / 'jogos_ativos.xlsx'
OUTPUT_STATS_CSV = ROOT / 'estatisticas_1st_half_playwright.csv'

STORAGE_FILE = ROOT / "sofascore_cookies.json"
PASTAS_LIMPAR = [ROOT / "excel_dados", ROOT / "json_dados"]

MAX_WORKERS = min(os.cpu_count() * 2, 10)
CHUNK_SIZE = 50
TIMEOUT = 6
NUMBER_OF_DAYS = 2
HEADLESS = True
LOG_LEVEL = logging.INFO

# toggle: if True keep same strict filter as before, else accept any event with player stats
APPLY_STRICT_TOURNAMENT_FILTER = True

# endpoints to download (only JSON)
ENDPOINTS = {
    'statistics': "https://www.sofascore.com/api/v1/event/{game_id}/statistics",
    'graph': "https://www.sofascore.com/api/v1/event/{game_id}/graph",
    'incidents': "https://www.sofascore.com/api/v1/event/{game_id}/incidents",
    'odds': "https://www.sofascore.com/api/v1/event/{game_id}/odds/1/all"
}

# logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Utilities ----------
def limpar_pastas(paths: List[Path]):
    for p in paths:
        try:
            if p.exists():
                for item in p.iterdir():
                    try:
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                        logging.info(f"Removed: {item}")
                    except Exception as e:
                        logging.warning(f"Failed remove {item}: {e}")
            else:
                logging.info(f"Folder not found: {p}")
        except Exception as e:
            logging.warning(f"Error processing folder {p}: {e}")

def remover_varios_arquivos(files: List[Path]):
    for f in files:
        try:
            if f.exists():
                f.unlink()
                logging.info(f"Removed old {f}")
            else:
                logging.info(f"File not found: {f}")
        except Exception as e:
            logging.warning(f"Failed remove file {f}: {e}")

def gerar_lista_de_datas(num_dias: int) -> List[str]:
    return [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(num_dias)]

def ler_planilha_flexivel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    suf = path.suffix.lower()
    if suf in ('.xlsx', '.xls'):
        return pd.read_excel(path, dtype=str)
    if suf in ('.csv', '.txt'):
        encs = ['utf-8','latin1','utf-16','cp1252']
        last = None
        for e in encs:
            try:
                return pd.read_csv(path, dtype=str, encoding=e, sep=None, engine='python')
            except Exception as ex:
                last = ex
        raise last
    # try magic
    with open(path,'rb') as f:
        start = f.read(8)
    if start.startswith(b'\x50\x4B\x03\x04'):
        return pd.read_excel(path, dtype=str)
    raise ValueError("Unsupported file type for: " + str(path))

def salvar_planilha_com_baixar(df: pd.DataFrame, path: Path):
    """Salva DataFrame garantindo que a coluna Baixar existe e tem valores apropriados"""
    if 'Baixar' not in df.columns:
        df['Baixar'] = '0'
    
    # Garantir que valores são strings
    df['Baixar'] = df['Baixar'].astype(str)
    
    # Salvar como Excel
    df.to_excel(path, index=False)

# ---------- Playwright-based collector ----------
def processar_evento(evento: Dict[str, Any], data_jogo: str) -> Dict[str, Any]:
    try:
        status_desc = str(evento.get("status", {}).get("description", "")).lower()
        status_type = str(evento.get("status", {}).get("type", "")).lower()
        return {
            'DIA_do_JOGO': data_jogo,
            'ID_Jogo': evento.get('id'),
            'CustomId': evento.get('customId'),
            'Torneio': evento.get('tournament', {}).get('name'),
            'Temporada': evento.get('season', {}).get('name'),
            'Time_Home': evento.get('homeTeam', {}).get('name'),
            'Time_Away': evento.get('awayTeam', {}).get('name'),
            'slug_Home': evento.get('homeTeam', {}).get('slug'),
            'slug_Away': evento.get('awayTeam', {}).get('slug'),
            'Placar_Home': evento.get('homeScore', {}).get('current', 0),
            'Placar_Away': evento.get('awayScore', {}).get('current', 0),
            'Status': status_desc,
            'Tipo_Status': status_type,
            'Inicio': datetime.fromtimestamp(evento['startTimestamp']).isoformat() if evento.get('startTimestamp') else None,
            'Atual_time': (datetime.fromtimestamp(evento.get('timestamp')).isoformat() if evento.get('timestamp') else (datetime.fromtimestamp(evento.get('statusTime', {}).get('timestamp')).isoformat() if evento.get('statusTime', {}).get('timestamp') else None)),
            'Injury_Time': f'+{evento.get("time", {}).get("injuryTime1", 0)}',
            'Baixar': '0'  # Default value
        }
    except Exception as e:
        logging.warning(f"Event process error: {e}")
        return None

async def coletar_eventos_playwright(numero_de_dias: int = NUMBER_OF_DAYS) -> List[Dict[str,Any]]:
    todos_eventos = []
    datas = gerar_lista_de_datas(numero_de_dias)
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0",
        "accept-language": "en-US,en;q=0.9"
    }

    async with async_playwright() as pw:
        request_ctx: APIRequestContext = await pw.request.new_context()
        async with AsyncCamoufox(headless=HEADLESS) as browser:
            context = await (browser.new_context(storage_state=str(STORAGE_FILE), viewport={"width":1920,"height":1080}) if STORAGE_FILE.exists() else browser.new_context(viewport={"width":1920,"height":1080}))
            try:
                for current_date in datas:
                    logging.info("Processing date: %s", current_date)
                    api_url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{current_date}"
                    eventos_brutos = []
                    try:
                        api_resp = await request_ctx.get(api_url, headers=headers, timeout=60000)
                        if api_resp.status == 200:
                            json_data = await api_resp.json()
                            eventos_brutos = json_data.get('events', [])
                            logging.info("API returned %d events for %s", len(eventos_brutos), current_date)
                        else:
                            logging.warning("API returned status %s for %s", api_resp.status, current_date)
                    except Exception:
                        logging.exception("Direct API request failed for %s", current_date)

                    passed = []
                    for evento in eventos_brutos:
                        unique = evento.get("tournament", {}).get("uniqueTournament", {})
                        has_perf_graph = unique.get("hasPerformanceGraphFeature", True)
                        has_player_stats = unique.get("hasEventPlayerStatistics", False)
                        cond = (not has_perf_graph and has_player_stats) if APPLY_STRICT_TOURNAMENT_FILTER else has_player_stats
                        if not cond:
                            continue
                        d = processar_evento(evento, current_date)
                        if d and d["Status"] not in ["not started","postponed","canceled","walkover","abandoned","retired"]:
                            passed.append(d)
                    logging.info("Date %s: total=%d, passed=%d", current_date, len(eventos_brutos), len(passed))
                    todos_eventos.extend(passed)
                # save storage_state
                try:
                    # create fresh context to save
                    tmp_ctx = await browser.new_context(viewport={"width":1920,"height":1080})
                    await tmp_ctx.storage_state(path=str(STORAGE_FILE))
                    await tmp_ctx.close()
                except Exception:
                    logging.debug("Failed to save storage_state (non-fatal)")
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
        await request_ctx.dispose()
    return todos_eventos

# ---------- Download endpoints JSON (Playwright request_ctx preferred) ----------
async def download_endpoint_with_playwright(request_ctx: APIRequestContext, url: str, timeout_ms: int = 60000):
    try:
        resp = await request_ctx.get(url, timeout=timeout_ms)
        status = resp.status
        if status == 200:
            try:
                return await resp.json(), status
            except Exception:
                txt = await resp.text()
                return txt, status
        return None, status
    except Exception as e:
        return None, 0

async def download_endpoint_aio(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, timeout=TIMEOUT, headers={"User-Agent":"Mozilla/5.0", "Referer":"https://www.sofascore.com/pt-pt/"}) as resp:
            status = resp.status
            if status == 200:
                try:
                    return await resp.json(), status
                except Exception:
                    return await resp.text(), status
            return None, status
    except Exception:
        return None, 0

async def baixar_apis_para_jogos(game_ids: List[str], request_ctx: APIRequestContext = None):
    """
    game_ids: list of game_id strings
    Saves JSON files under BASE/<game_id>/<endpoint>.json
    Returns list of dicts with summary per game.
    """
    resumo = []
    sem = asyncio.Semaphore(10)  # limit parallel downloads
    async with aiohttp.ClientSession() as session:
        async def _baixar(game_id: str):
            async with sem:
                endpoints_downloaded = []
                endpoints_status = {}
                dir_path = BASE / str(game_id)
                dir_path.mkdir(exist_ok=True, parents=True)
                for name, template in ENDPOINTS.items():
                    url = template.format(game_id=game_id)
                    # try playwright first if available
                    dados, status = (await download_endpoint_with_playwright(request_ctx, url)) if request_ctx is not None else (None, 0)
                    if status != 200:
                        # fallback aiohttp (with 2 retries)
                        for attempt in range(2):
                            dados, status = await download_endpoint_aio(session, url)
                            if status == 200:
                                break
                            await asyncio.sleep(0.5 * (2 ** attempt))
                    endpoints_status[name] = status
                    if status == 200 and dados is not None:
                        file_path = dir_path / f"{name}.json"
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "data": dados}, f, ensure_ascii=False, indent=2)
                            endpoints_downloaded.append(name)
                        except Exception:
                            logging.exception("Failed write json for %s %s", game_id, name)
                resumo.append({"game_id": game_id, "downloaded": endpoints_downloaded, "status": "baixado" if endpoints_downloaded else "tentado", "endpoints_status": endpoints_status})
        tasks = [_baixar(g) for g in game_ids]
        await asyncio.gather(*tasks)
    return resumo

# ---------- Statistics processing (1ST half) ----------
def calcular_hash_arquivo(path: Path):
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return None

def processar_statistics_1st_from_json_pair(pair: Tuple[str, Path]):
    gid, json_path = pair
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None  # skip

    data_field = j.get("data", {})
    if not data_field:
        return None

    statistics = data_field.get("statistics", [])
    if not statistics:
        return None

    out = {"ID_Jogo": gid}
    for group in statistics:
        if group.get("period") != "1ST":
            continue
        groups = group.get("groups", [])
        for g in groups:
            items = g.get("statisticsItems", [])
            for it in items:
                name = it.get("name")
                if not name:
                    continue
                key_base = "".join(ch if (ch.isalnum() or ch == ' ') else '_' for ch in name).strip().replace(" ", "_")
                home_val = it.get("homeValue")
                away_val = it.get("awayValue")
                # fallback to string home/away fields
                if home_val is None:
                    hv = it.get("home")
                else:
                    hv = home_val
                if away_val is None:
                    av = it.get("away")
                else:
                    av = away_val
                out[f"{key_base}_Casa"] = hv
                out[f"{key_base}_Fora"] = av
    if len(out) <= 1:
        return None
    return out

def Processar_estatisticas_paralelo_from_jogos_xlsx():
    try:
        if not JOGOS_XLSX.exists():
            logging.error("jogos_ativos.xlsx not found at %s", JOGOS_XLSX)
            return None

        df_jogos = ler_planilha_flexivel(JOGOS_XLSX)
        if df_jogos.empty:
            logging.error("jogos_ativos.xlsx is empty")
            return None
            
        # detect status column
        status_col = None
        if 'Tipo_Status' in df_jogos.columns:
            status_col = 'Tipo_Status'
        else:
            cols = [c for c in df_jogos.columns if 'status' in c.lower()]
            status_col = cols[0] if cols else None

        # select finished or inprogress
        finished = df_jogos[df_jogos[status_col].fillna('').astype(str).str.lower() == 'finished'] if status_col else pd.DataFrame()
        inprog = df_jogos[df_jogos[status_col].fillna('').astype(str).str.lower() == 'inprogress'] if status_col else pd.DataFrame()
        ids = set(finished['ID_Jogo'].dropna().astype(str).tolist()) | set(inprog['ID_Jogo'].dropna().astype(str).tolist())
        logging.info("IDs to consider for stats: %d", len(ids))

        # Build candidate list from BASE/<id>/statistics.json
        candidate_pairs = []
        for gid in ids:
            path = BASE / str(gid) / "statistics.json"
            if path.exists():
                candidate_pairs.append((gid, path))
        logging.info("Found %d statistics.json files matching IDs", len(candidate_pairs))
        if not candidate_pairs:
            logging.info("No statistics files to process.")
            return None

        # load registry
        registry = {}
        if REGISTRY_PATH.exists():
            try:
                registry = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
            except Exception:
                registry = {}

        # decide which to process (hash changed or inprogress)
        arquivos_para = []
        hash_updates = {}
        for gid, path in candidate_pairs:
            hash_val = calcular_hash_arquivo(path)
            reg = registry.get(gid)
            if gid in set(inprog['ID_Jogo'].astype(str).tolist()):
                arquivos_para.append((gid, path))
            elif not reg:
                arquivos_para.append((gid, path))
            else:
                if reg.get('hash') != hash_val:
                    arquivos_para.append((gid, path))
                # else skip

        logging.info("To process (after registry check): %d files", len(arquivos_para))
        if not arquivos_para:
            logging.info("Nothing to process after registry check.")
            return None

        # parallel map
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(processar_statistics_1st_from_json_pair, pair): pair for pair in arquivos_para}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                pair = futures[fut]
                if res:
                    results.append(res)
                    # store hash update
                    h = calcular_hash_arquivo(pair[1])
                    if h:
                        hash_updates[pair[0]] = h

        if not results:
            logging.info("No statistics extracted.")
            return None

        df_new = pd.DataFrame(results)
        # numeric clean
        for col in df_new.columns:
            if col != 'ID_Jogo':
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

        # load existing output csv and upsert
        df_exist = pd.DataFrame()
        if Path(OUTPUT_STATS_CSV).exists():
            try:
                df_exist = pd.read_csv(OUTPUT_STATS_CSV, dtype=str)
            except Exception:
                df_exist = pd.DataFrame()

        if not df_exist.empty:
            try:
                # ensure numeric cols cast
                for c in df_exist.columns:
                    if c != 'ID_Jogo':
                        df_exist[c] = pd.to_numeric(df_exist[c], errors='coerce')
                df_final = pd.concat([df_exist, df_new], ignore_index=True)
                df_final = df_final.drop_duplicates(subset=['ID_Jogo'], keep='last')
            except Exception:
                df_final = df_new
        else:
            df_final = df_new

        # save CSV with rotation
        try:
            if Path(OUTPUT_STATS_CSV).exists():
                bak1 = Path(str(OUTPUT_STATS_CSV)+".bak1")
                if bak1.exists():
                    bak1.unlink()
                Path(OUTPUT_STATS_CSV).replace(bak1)
        except Exception:
            pass
        df_final.to_csv(OUTPUT_STATS_CSV, index=False, encoding='utf-8')
        logging.info("Saved stats CSV: %s (%d rows)", OUTPUT_STATS_CSV, len(df_final))

        # update registry
        if hash_updates:
            now_iso = datetime.utcnow().isoformat() + "Z"
            for gid, h in hash_updates.items():
                registry[gid] = {"hash": h, "updated_at": now_iso}
            try:
                REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logging.exception("Failed save registry")

        return df_final

    except Exception as e:
        logging.exception("Fatal error in stats processing: %s", e)
        return None

# ---------- Main pipeline ----------
async def main_pipeline():
    logging.info("Starting pipeline: collect events -> save -> download APIs -> process stats")
    
    # NÃO limpar pastas - manter ficheiros criados anteriormente
    # limpar_pastas([p for p in PASTAS_LIMPAR])
    # remover_varios_arquivos([JOGOS_XLSX])

    # 1) collect events
    eventos = await coletar_eventos_playwright(NUMBER_OF_DAYS)
    if eventos:
        df_novos = pd.DataFrame(eventos)
        
        # calcula minutos e parte (re-using simple approach)
        def calcular_minutos_row(row):
            try:
                status = str(row.get("Status", "")).lower()

                # Para jogos não iniciados
                if status in ["not started", "canceled", "postponed"]:
                    return "00:00"

                # Para intervalo
                if status == "halftime":
                    return "45:00"

                # Para jogos finalizados
                if status in ["ended", "finished", "after penalties", "after extra time"]:
                    return "90:00"

                # Para jogos em andamento
                if pd.notnull(row["Atual_time"]):
                    atual_inicio = pd.to_datetime(row["Atual_time"], utc=True)
                    now = pd.to_datetime('now', utc=True)
                    diff = now - atual_inicio

                    # Se o Status iniciar com "2nd", adicionar 45 minutos
                    if status.startswith("2nd"):
                        diff += timedelta(minutes=45)

                    if diff.total_seconds() < 0:
                        diff = timedelta(0)

                    minutos = int(diff.total_seconds() // 60)
                    segundos = int(diff.total_seconds() % 60)

                    # Limitar a 90 minutos para jogos em andamento
                    if minutos > 90:
                        minutos = 90
                        segundos = 0

                    return f"{minutos:02d}:{segundos:02d}"
                else:
                    return "00:00"
            except Exception as e:
                print(f"Erro ao calcular minutos: {e}")
                return "00:00"
                
        df_novos['Minutos_jogo'] = df_novos.apply(calcular_minutos_row, axis=1)
        
        # Verificar se arquivo já existe e mesclar sem duplicados
        if JOGOS_XLSX.exists():
            df_existente = ler_planilha_flexivel(JOGOS_XLSX)
            if not df_existente.empty:
                # Garantir que ID_Jogo é string para comparação
                df_existente['ID_Jogo'] = df_existente['ID_Jogo'].astype(str)
                df_novos['ID_Jogo'] = df_novos['ID_Jogo'].astype(str)
                
                # Manter valores de 'Baixar' dos jogos existentes
                if 'Baixar' in df_existente.columns:
                    # Criar dicionário de mapeamento ID_Jogo -> Baixar
                    baixar_map = df_existente.set_index('ID_Jogo')['Baixar'].to_dict()
                    
                    # Atualizar valores de Baixar nos novos dados com base no existente
                    df_novos['Baixar'] = df_novos['ID_Jogo'].map(baixar_map).fillna('0')
                
                # Remover duplicados mantendo os mais recentes (baseado na ordem dos novos dados)
                df_final = pd.concat([df_existente, df_novos], ignore_index=True)
                df_final = df_final.drop_duplicates(subset=['ID_Jogo'], keep='last')
            else:
                df_final = df_novos
        else:
            df_final = df_novos
        
        # Salvar arquivo atualizado
        salvar_planilha_com_baixar(df_final, JOGOS_XLSX)
        logging.info("Saved/updated games xlsx: %s (Total: %d jogos)", JOGOS_XLSX, len(df_final))
    else:
        # Se não houver eventos, apenas garantir que o arquivo existe
        if not JOGOS_XLSX.exists():
            df_vazio = pd.DataFrame(columns=['ID_Jogo', 'DIA_do_JOGO', 'Baixar'])
            salvar_planilha_com_baixar(df_vazio, JOGOS_XLSX)
            logging.info("Created empty games xlsx: %s", JOGOS_XLSX)
        else:
            logging.info("No new events found, keeping existing xlsx")

    # 2) read jogos_ativos.xlsx and build game list for download
    try:
        df_j = ler_planilha_flexivel(JOGOS_XLSX)
        if df_j.empty:
            logging.info("jogos_ativos.xlsx is empty, skipping download")
            return
    except Exception as e:
        logging.error("Failed open jogos xlsx: %s", e)
        return
        
    # Garantir que colunas necessárias existem
    if 'ID_Jogo' not in df_j.columns:
        logging.error("Column ID_Jogo not found in jogos_ativos.xlsx")
        return
        
    if 'Tipo_Status' not in df_j.columns:
        logging.error("Column Tipo_Status not found in jogos_ativos.xlsx")
        return
        
    if 'Baixar' not in df_j.columns:
        df_j['Baixar'] = '0'
        salvar_planilha_com_baixar(df_j, JOGOS_XLSX)

    # Lógica de download baseada em Tipo_Status e coluna Baixar
    game_ids_to_download = []
    jogos_para_atualizar = []
    
    for idx, row in df_j.iterrows():
        game_id = str(row['ID_Jogo'])
        tipo_status = str(row.get('Tipo_Status', '')).lower()
        baixar = str(row.get('Baixar', '0'))
        
        if tipo_status == 'inprogress':
            # Jogos em andamento: sempre baixar
            game_ids_to_download.append(game_id)
            if baixar != '0':
                # Atualizar coluna Baixar para 0 (para indicar que precisa baixar)
                df_j.at[idx, 'Baixar'] = '0'
                jogos_para_atualizar.append(idx)
                
        elif tipo_status == 'finished':
            # Jogos finalizados: baixar apenas se Baixar == '0'
            if baixar == '0':
                game_ids_to_download.append(game_id)
                # Marcar como 1 para indicar que será baixado
                df_j.at[idx, 'Baixar'] = '1'
                jogos_para_atualizar.append(idx)
        else:
            # Outros status: não baixar
            pass
    
    # Atualizar arquivo se necessário
    if jogos_para_atualizar:
        salvar_planilha_com_baixar(df_j, JOGOS_XLSX)
        logging.info("Updated Baixar column for %d jogos", len(jogos_para_atualizar))
    
    logging.info("Found %d game IDs to download", len(game_ids_to_download))
    
    if game_ids_to_download:
        # 3) download APIs JSON for these game_ids using Playwright request_ctx
        async with async_playwright() as pw:
            request_ctx = await pw.request.new_context()
            resumo = await baixar_apis_para_jogos(game_ids_to_download, request_ctx=request_ctx)
            await request_ctx.dispose()
        logging.info("APIs download finished: %d results", len(resumo))
        
        # Atualizar coluna Baixar para jogos que foram realmente baixados
        jogos_baixados = [r['game_id'] for r in resumo if r['downloaded']]
        for idx, row in df_j.iterrows():
            if str(row['ID_Jogo']) in jogos_baixados and str(row.get('Tipo_Status', '')).lower() == 'finished':
                df_j.at[idx, 'Baixar'] = '1'
        
        salvar_planilha_com_baixar(df_j, JOGOS_XLSX)
        logging.info("Updated Baixar column after download")
    else:
        logging.info("No games to download based on status and Baixar column")

    # 4) process statistics from downloaded JSON files
    df_stats = Processar_estatisticas_paralelo_from_jogos_xlsx()
    if df_stats is not None:
        logging.info("Statistics processing completed: %d rows", len(df_stats))
    else:
        logging.warning("Statistics processing returned no results")

if __name__ == "__main__":
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.exception("Fatal error in pipeline")








####################################################
# ####################################################
# ####################################################
# # End of collect_and_process.py
####################################################      



