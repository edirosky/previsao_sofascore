# @title reprocess_goals.py — Reprocessamento de golos com registro de hash (para GitHub Codespaces / VSCode)
"""
Executar:
  python scripts/reprocess_goals.py

O script:
- Lê IDs válidos de ./data/jogos_ativos.xlsx
- Procura por incidents.json em ./data/playwright_jsons_full/<ID>/incidents.json
- Usa registry de hashes para evitar reprocessamento
- Gera CSV com evolução do placar no formato: "1-0 (29') → 2-0 (38') → ..."
- Salva registry atualizado em ./data/playwright_jsons_full/golos_ordenados_hash.json
- Salva CSV em ./data/golos_placar_ordenados_playwright.csv
"""

from pathlib import Path
import json
import hashlib
import re
import traceback
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
import concurrent.futures

import pandas as pd
from tqdm.auto import tqdm

# -------------------------
# Config / Paths
# -------------------------
ROOT = Path("./data")
BASE = ROOT / "playwright_jsons_full"
ROOT.mkdir(parents=True, exist_ok=True)
BASE.mkdir(parents=True, exist_ok=True)

JOGOS_ATIVOS_XLSX = ROOT / "jogos_ativos.xlsx"                    # fonte de IDs válidos
GOLOS_CSV_PATH = ROOT / "golos_placar_ordenados_playwright.csv"  # saída final
HASH_REGISTRY_PATH = BASE / "golos_ordenados_hash.json"          # registry de hashes
INCIDENT_FILENAME = "incidents.json"

MAX_WORKERS = 8

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------
# Helpers
# -------------------------
def _hash_file(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def parse_minute_int(value) -> int:
    """Extrai minuto para ordenação, ex: '45+2' -> 45, '29' -> 29."""
    try:
        s = str(value)
        m = re.search(r'\d+', s)
        return int(m.group()) if m else 0
    except Exception:
        return 0


def minute_display(value) -> str:
    """Formata minuto para exibição: '45+2' -> \"45+2'\", 29 -> \"29'\"."""
    if value is None:
        return "0'"
    s = str(value).strip()
    if s == "":
        return "0'"
    if s.endswith("'") or s.endswith("’"):
        return s
    return f"{s}'"


def load_allowed_ids_from_xlsx(xlsx_path: Path) -> set:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Arquivo de jogos não encontrado: {xlsx_path}")
    df = pd.read_excel(xlsx_path, dtype=str)
    if "ID_Jogo" not in df.columns:
        # tentar colunas alternativas (case-insensitive)
        alt = [c for c in df.columns if c.lower() in ("id_jogo", "id", "game_id")]
        if alt:
            id_col = alt[0]
        else:
            raise KeyError("Coluna 'ID_Jogo' não encontrada em jogos_ativos.xlsx")
    else:
        id_col = "ID_Jogo"
    ids = set(df[id_col].dropna().astype(str).str.strip().tolist())
    return ids


# -------------------------
# Processamento de um jogo (incidents.json)
# -------------------------
def processar_arquivo_incidents(gid_and_path: Tuple[str, Path]) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Processa um incidents.json e retorna (gid, summary_dict, hash) ou (gid, None, None) em erro/sem dados.
    summary_dict contém:
      - ID_Jogo, minutos Golos_Casa, minutos Golos_Fora, PLACAR HT, PLACAR FT,
        total_golos_casa, total_golos_fora, quantidade_gols, evolução do Placar, timestamp_processamento
    """
    gid, arquivo = gid_and_path
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.debug("Falha a ler %s: %s", arquivo, e)
        return gid, None, None

    incidents = data.get("data", {}).get("incidents", []) or []
    if not incidents:
        return gid, None, None

    # ordenar por minuto (extrair número)
    try:
        incidents_sorted = sorted(incidents, key=lambda x: parse_minute_int(x.get("time", 0)))
    except Exception:
        incidents_sorted = incidents

    home_goals = []
    away_goals = []
    home_running = 0
    away_running = 0
    evolucao_events = []
    home_score_ht = None
    away_score_ht = None
    home_score_ft = None
    away_score_ft = None

    for incident in incidents_sorted:
        incident_type = incident.get("incidentType", "")
        # periodos que podem trazer HT/FT
        if incident_type == "period":
            time_val = incident.get("time", 0)
            text_val = str(incident.get("text", "")).upper()
            home_score = incident.get("homeScore", None)
            away_score = incident.get("awayScore", None)
            if text_val == "HT" or parse_minute_int(time_val) == 45:
                try:
                    if home_score is not None:
                        home_score_ht = int(home_score)
                    if away_score is not None:
                        away_score_ht = int(away_score)
                except Exception:
                    pass
            if text_val == "FT" or parse_minute_int(time_val) >= 90:
                try:
                    if home_score is not None:
                        home_score_ft = int(home_score)
                    if away_score is not None:
                        away_score_ft = int(away_score)
                except Exception:
                    pass

        # goals
        if incident_type == "goal":
            time_val = incident.get("time", 0)
            minute_num = parse_minute_int(time_val)
            minute_txt = minute_display(time_val)
            is_home = incident.get("isHome", False)

            if is_home:
                home_goals.append(minute_num)
                home_running += 1
            else:
                away_goals.append(minute_num)
                away_running += 1

            evolucao_events.append(f"{home_running}-{away_running} ({minute_txt})")

    # defaults HT/FT if None
    if home_score_ht is None:
        home_score_ht = 0
    if away_score_ht is None:
        away_score_ht = 0
    if home_score_ft is None:
        home_score_ft = home_running
    if away_score_ft is None:
        away_score_ft = away_running

    summary = {
        "ID_Jogo": str(gid),
        "minutos_Golos_Casa": ",".join(map(str, home_goals)) if home_goals else "",
        "minutos_Golos_Fora": ",".join(map(str, away_goals)) if away_goals else "",
        "PLACAR_HT": f"{home_score_ht}-{away_score_ht}",
        "PLACAR_FT": f"{home_score_ft}-{away_score_ft}",
        "total_golos_casa": len(home_goals),
        "total_golos_fora": len(away_goals),
        "quantidade_gols": len(home_goals) + len(away_goals),
        "evolucao_do_Placar": " → ".join(evolucao_events) if evolucao_events else "",
        "timestamp_processamento": datetime.utcnow().isoformat() + "Z",
    }

    current_hash = _hash_file(arquivo)
    return gid, summary, current_hash


# -------------------------
# Main reprocess function
# -------------------------
def reprocessar_golos_com_registro_efetivo() -> Dict:
    try:
        logging.info("🔄 Iniciando reprocessamento de golos com registro (ROOT=%s)", ROOT)

        # carregar ids permitidos a partir do jogos_ativos.xlsx
        allowed_ids = load_allowed_ids_from_xlsx(JOGOS_ATIVOS_XLSX)
        logging.info("📋 IDs permitidos carregados: %d", len(allowed_ids))

        # carregar registry de hash
        if HASH_REGISTRY_PATH.exists():
            try:
                registry = json.loads(HASH_REGISTRY_PATH.read_text(encoding="utf-8"))
                logging.info("📦 Registry carregado: %d entradas", len(registry))
            except Exception:
                logging.warning("⚠️ Registry inválido — recriando novo")
                registry = {}
        else:
            logging.info("📦 Registry não encontrado — será criado novo")
            registry = {}

        # procurar incidents.json
        arquivos = list(BASE.rglob(INCIDENT_FILENAME))
        logging.info("🔍 Arquivos incidents.json encontrados (total): %d", len(arquivos))

        # map gid -> path (apenas IDs permitidos)
        gid_to_path = {}
        for arquivo in arquivos:
            gid = arquivo.parent.name
            if gid in allowed_ids:
                gid_to_path[gid] = arquivo

        logging.info("🔎 Arquivos correspondentes a IDs permitidos: %d", len(gid_to_path))

        # decidir quais precisam processar (novos / hash change / inexistent skip)
        to_process = {}
        skipped = 0
        novo = 0
        atualizado = 0

        for gid, path in gid_to_path.items():
            h = _hash_file(path)
            if h is None:
                # leitura falhou, marcar skip
                skipped += 1
                continue
            prev = registry.get(gid)
            if not prev:
                to_process[gid] = path
                novo += 1
            else:
                if prev == h:
                    skipped += 1
                else:
                    to_process[gid] = path
                    atualizado += 1

        logging.info("🎯 A processar: %d | novos: %d | atualizados: %d | pulados: %d",
                     len(to_process), novo, atualizado, skipped)

        if not to_process:
            logging.info("✅ Nada para processar. Saindo.")
            return {"novos": 0, "atualizados": 0, "pulados": skipped, "total_arquivos": len(arquivos)}

        # processamento paralelo
        trabalhos = list(to_process.items())
        resultados = []
        novos_hashes: Dict[str, str] = {}
        stats = {"sucesso": 0, "erros": 0, "golos_total": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(processar_arquivo_incidents, t): t[0] for t in trabalhos}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="📊 Processando jogos", unit="jogo"):
                gid = futures[future]
                try:
                    gid_r, summary, current_hash = future.result()
                except Exception:
                    logging.exception("Erro no future para gid %s", gid)
                    gid_r, summary, current_hash = gid, None, None

                if summary:
                    resultados.append(summary)
                    stats["sucesso"] += 1
                    stats["golos_total"] += int(summary.get("quantidade_gols", 0) or 0)
                    if current_hash:
                        novos_hashes[gid_r] = current_hash
                else:
                    stats["erros"] += 1

        logging.info("✅ Processamento paralelo concluído: %d processados com sucesso, %d erros, %d golos totais",
                     stats["sucesso"], stats["erros"], stats["golos_total"])

        # combinar e salvar CSV final (upsert)
        if resultados:
            df_novos = pd.DataFrame(resultados)

            # ler arquivo existente se existir e fazer upsert por ID_Jogo
            df_existente = pd.DataFrame()
            if GOLOS_CSV_PATH.exists():
                try:
                    df_existente = pd.read_csv(GOLOS_CSV_PATH, dtype={"ID_Jogo": str})
                except Exception:
                    logging.warning("⚠️ Falha ao ler CSV existente, será recriado.")

            if not df_existente.empty:
                ids_novos = set(df_novos["ID_Jogo"].astype(str).tolist())
                df_existente = df_existente[~df_existente["ID_Jogo"].astype(str).isin(ids_novos)]
                df_final = pd.concat([df_existente, df_novos], ignore_index=True)
            else:
                df_final = df_novos

            # remover duplicados e ordenar por ID numericamente quando possível
            df_final = df_final.drop_duplicates(subset=["ID_Jogo"], keep="last")
            def sort_key(gid_val):
                try:
                    return int("".join(filter(str.isdigit, str(gid_val))))
                except Exception:
                    return float("inf")
            df_final["_sort"] = df_final["ID_Jogo"].apply(sort_key)
            df_final = df_final.sort_values("_sort").reset_index(drop=True).drop(columns=["_sort"])

            # salvar CSV (utf-8-sig para compatibilidade)
            df_final.to_csv(GOLOS_CSV_PATH, index=False, encoding="utf-8-sig")
            logging.info("💾 CSV salvo: %s (%d registros)", GOLOS_CSV_PATH, len(df_final))
        else:
            logging.info("⚠️ Nenhum resultado válido para salvar.")

        # atualizar registry
        if novos_hashes:
            registry.update(novos_hashes)
            try:
                HASH_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
                logging.info("📦 Registry atualizado: %d entradas", len(novos_hashes))
            except Exception:
                logging.exception("Erro ao salvar registry")

        # relatório final
        logging.info("🎉 Reprocessamento concluído: processados=%d, erros=%d, golos_total=%d, novos=%d, atualizados=%d, pulados=%d",
                     stats["sucesso"], stats["erros"], stats["golos_total"], novo, atualizado, skipped)

        return {"processados": stats["sucesso"], "erros": stats["erros"], "total_golos": stats["golos_total"],
                "novos": novo, "atualizados": atualizado, "pulados": skipped}

    except Exception:
        logging.exception("ERRO FATAL no reprocessamento")
        return {"processados": 0, "erros": 1, "total_golos": 0, "novos": 0, "atualizados": 0, "pulados": 0}


# -------------------------
# Execução
# -------------------------
if __name__ == "__main__":
    result = reprocessar_golos_com_registro_efetivo()
    logging.info("Resultado final: %s", result)
