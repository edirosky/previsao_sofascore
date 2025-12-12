# -*- coding: utf-8 -*-
"""
Script para criar o arquivo incidentes_estatisticas_geral.csv
- Faz merge de jogos, estatísticas e golos
- Normaliza colunas
- Converte colunas *_Casa e *_Fora para inteiros
- Salva o CSV final
"""

import os
import logging
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import numpy as np

def garantir_csv_jogos(base_dir: str, jogos_filename: str, base_excel: Optional[str] = "jogos_ativos.xlsx"):
    """Garante que o CSV de jogos exista, criando-o se necessário."""
    path = os.path.join(base_dir, jogos_filename)
    if not os.path.isfile(path):
        print(f"⚠️ Arquivo '{jogos_filename}' não encontrado. Criando automaticamente...")
        excel_path = os.path.join(base_dir, base_excel) if base_excel else None
        if excel_path and os.path.isfile(excel_path):
            df = pd.read_excel(excel_path)
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ '{jogos_filename}' criado a partir de '{base_excel}'")
        else:
            df = pd.DataFrame(columns=["ID_Jogo"])
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ '{jogos_filename}' criado vazio com coluna 'ID_Jogo'")
    return path

def Criar_incidentes_estatisticas_geral(
    base_dir: str = "/workspaces/previsao_sofascore/scripts/data",
    jogos_filename: str = "estatisticas_e_golos.csv",
    estatisticas_filename: str = "estatisticas_1st_half_playwright.csv",
    golos_filename: str = "golos_placar_ordenados_playwright.csv",
    output_filename: str = "incidentes_estatisticas_geral.csv",
    save: bool = True,
    overwrite: bool = True,
    logging_level: int = logging.INFO,
    log_to_file: bool = False,
    log_file_path: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Lê CSVs de jogos, estatísticas e golos, realiza merges, normaliza colunas,
    converte colunas *_Casa e *_Fora para inteiros e salva o CSV final.
    
    Retorna:
        final_df: pd.DataFrame com dados processados
        metadata: dict com informações do processamento
    """
    # ---------- Configurar logger ----------
    logger_name = "Criar_incidentes_estatisticas_geral"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    logger.setLevel(logging_level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        if not log_file_path:
            log_file_path = os.path.join(base_dir, f"{logger_name}.log")
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(stream_formatter)
        logger.addHandler(file_handler)

    logger.info("🔄 Iniciando criação do arquivo incidentes_estatisticas_geral.csv")

    # ---------- Garantir que o CSV de jogos exista ----------
    jogos_path = garantir_csv_jogos(base_dir, jogos_filename)

    # ---------- Preparar paths ----------
    estatisticas_path = os.path.join(base_dir, estatisticas_filename)
    golos_path = os.path.join(base_dir, golos_filename)
    output_path = os.path.join(base_dir, output_filename)

    # Verificar existência dos outros arquivos
    missing_files = [p for p in (estatisticas_path, golos_path) if not os.path.isfile(p)]
    if missing_files:
        logger.error("Ficheiros em falta: %s", missing_files)
        raise FileNotFoundError(f"Ficheiros em falta: {missing_files}")

    # ---------- Carregar CSVs ----------
    try:
        jogos_df = pd.read_csv(jogos_path)
        estatisticas_df = pd.read_csv(estatisticas_path)
        golos_df = pd.read_csv(golos_path)
    except Exception as e:
        logger.exception("Erro ao carregar CSVs: %s", e)
        raise

    logger.info("✅ CSVs carregados com sucesso.")

    # ---------- Normalizar nomes de colunas ----------
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
        return df

    jogos_df = _normalize_columns(jogos_df)
    estatisticas_df = _normalize_columns(estatisticas_df)
    golos_df = _normalize_columns(golos_df)

    key = "ID_Jogo"
    for name, df in [("jogos_df", jogos_df), ("estatisticas_df", estatisticas_df), ("golos_df", golos_df)]:
        if key not in df.columns:
            logger.error("Coluna de chave '%s' não encontrada em %s", key, name)
            raise KeyError(f"Coluna de chave '{key}' não encontrada em {name}")

    # ---------- Merge ----------
    merged_df = jogos_df.merge(estatisticas_df, on=key, how="left", suffixes=("", "_est"))
    final_df = merged_df.merge(golos_df, on=key, how="left", suffixes=("", "_golos"))

    # ---------- Ordenar colunas ----------
    colunas_base = [
        'ID_Jogo','Torneio','ID_Torneio','ID_Torneio_Unico','Torneio_Unico_Nome',
        'Temporada','ID_Temporada','Time_Home','ID_Time_Home','Time_Away',
        'ID_Time_Away','Placar_Home','Placar_Away','Status','Inicio','Atual_time',
        'Injury_Time','Tipo_Status','Data_Jogo','Has_HeatMap','Has_Statistics',
        'Minutos_jogo','Hash','status_estatisticas','status_incidentes','status_graph'
    ]
    colunas_golos = [
        'minutos Golos_Casa','minutos Golos_Fora','PLACAR HT','PLACAR FT',
        'evolução do Placar','total_gols_casa','total_gols_fora','quantidade_gols'
    ]
    colunas_estatisticas = [c for c in estatisticas_df.columns if c != key]

    colunas_finais = list(dict.fromkeys(
        [c for c in colunas_base if c in final_df.columns] +
        [c for c in colunas_golos if c in final_df.columns] +
        colunas_estatisticas +
        [c for c in final_df.columns if c not in colunas_base + colunas_golos + colunas_estatisticas]
    ))

    final_df = final_df.loc[:, colunas_finais]

    # ---------- Converter colunas *_Casa e *_Fora para inteiros ----------
    casa_cols = [c for c in final_df.columns if c.endswith('_Casa')]
    fora_cols = [c for c in final_df.columns if c.endswith('_Fora')]
    for col in casa_cols + fora_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)

    # ---------- Salvar CSV final ----------
    metadata = {
        'arquivo': output_filename,
        'data_criacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_jogos': final_df.shape[0],
        'total_colunas': final_df.shape[1],
        'fontes': [jogos_filename, estatisticas_filename, golos_filename],
        'base_dir': base_dir
    }

    if save:
        if os.path.exists(output_path) and not overwrite:
            logger.error("Arquivo já existe e overwrite=False: %s", output_path)
            raise FileExistsError(f"{output_path} já existe e overwrite=False")
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info("💾 CSV salvo: %s (%d linhas x %d colunas)", output_path, final_df.shape[0], final_df.shape[1])

    return final_df, metadata


if __name__ == "__main__":
    BASE_DIR = "/workspaces/previsao_sofascore/data"
    JOGOS_FILE = "estatisticas_e_golos.csv"

    df, meta = Criar_incidentes_estatisticas_geral(
        base_dir=BASE_DIR,
        jogos_filename=JOGOS_FILE,
        estatisticas_filename="estatisticas_1st_half_playwright.csv",
        golos_filename="golos_placar_ordenados_playwright.csv"
    )

    print(f"\n🎯 Metadata final: {meta}")
