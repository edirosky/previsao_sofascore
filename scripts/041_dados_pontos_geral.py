# @title PONTOS PARA GRAFICOS — recolha completa com métricas adicionais
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
import sys
from scipy import stats

# Configurar para evitar exibir warnings desnecessários
import warnings
warnings.filterwarnings('ignore')

# Configurações de caminhos
BASE = Path('/workspaces/previsao_sofascore/data')
JSONS_DIR = BASE / 'playwright_jsons_full'
TODAY_FOLDER = BASE
TODAY_FOLDER.mkdir(parents=True, exist_ok=True)

def Coleta_pontos_momentum_completa(
    ids_source_file: str = 'df_previsoes_sim_concatenado.csv',
    out_filename: str = 'dados_pontos_geral_completo.csv',
    minute_limit: int = 90,
    save: bool = True,
    verbose: bool = True
):
    """
    Recolhe gráficos de pontos para TODOS os jogos com métricas adicionais de ritmo
    Usa JSONs já baixados na pasta playwright_jsons_full no formato: {id_jogo}/graph.json
    Saída: dados_pontos_geral_completo.csv com colunas adicionais de métricas
    """
    try:
        minute_cols_str = [str(i) for i in range(1, minute_limit+1)]

        # ---------- helpers ----------
        def _normalize_id_col(df, col='ID_Jogo'):
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
            return df

        def _col_to_min_digits(s):
            found = re.findall(r'\d{1,3}', str(s))
            return found[0] if found else None

        def _collapse_minute_columns(df, minute_limit):
            df = df.copy()
            col_map = {}
            for c in df.columns:
                m = _col_to_min_digits(c)
                if m is None:
                    continue
                try:
                    mn = int(m)
                except:
                    continue
                if 1 <= mn <= minute_limit:
                    col_map.setdefault(str(mn), []).append(c)

            for m in range(1, minute_limit+1):
                m_str = str(m)
                candidates = col_map.get(m_str, [])
                if not candidates:
                    df[m_str] = 0.0
                    continue

                cand_df = df[candidates].replace('', np.nan).astype(float)
                cand_nonzero = cand_df.replace(0, np.nan)
                try:
                    chosen = cand_nonzero.bfill(axis=1).iloc[:, 0]
                except Exception:
                    chosen = cand_nonzero.iloc[:, 0]
                if chosen.isna().all():
                    try:
                        chosen = cand_df.bfill(axis=1).iloc[:, 0]
                    except Exception:
                        chosen = cand_df.iloc[:, 0]
                chosen = chosen.fillna(0).astype(float)
                df[m_str] = chosen

            to_drop = []
            for lst in col_map.values():
                for c in lst:
                    if str(c) not in minute_cols_str:
                        to_drop.append(c)
            to_drop = [c for c in set(to_drop) if c in df.columns]
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
            return df

        def calcular_metricas_ritmo(df_minutes):
            """
            Calcula métricas adicionais de ritmo do jogo
            df_minutes: DataFrame com valores por minuto (1 a 90)
            """
            metrics = {}
            
            if df_minutes.empty:
                return metrics
            
            # Converter para array numpy
            valores = df_minutes.values.flatten()
            
            # 1. INTENSIDADE BÁSICA
            metrics['intensidade_total'] = np.sum(np.abs(valores))
            metrics['intensidade_media'] = np.mean(np.abs(valores))
            metrics['intensidade_max'] = np.max(np.abs(valores))
            
            # 2. VARIABILIDADE
            metrics['variabilidade'] = np.std(valores)
            metrics['amplitude'] = np.max(valores) - np.min(valores)
            
            # 3. TENDÊNCIAS POR TEMPO
            # Primeiro tempo (1-45)
            primeiro_tempo = valores[:45] if len(valores) >= 45 else valores
            metrics['intensidade_1t'] = np.mean(np.abs(primeiro_tempo)) if len(primeiro_tempo) > 0 else 0
            metrics['dominio_1t'] = np.mean(primeiro_tempo) if len(primeiro_tempo) > 0 else 0
            
            # Segundo tempo (46-90)
            segundo_tempo = valores[45:] if len(valores) > 45 else []
            metrics['intensidade_2t'] = np.mean(np.abs(segundo_tempo)) if len(segundo_tempo) > 0 else 0
            metrics['dominio_2t'] = np.mean(segundo_tempo) if len(segundo_tempo) > 0 else 0
            
            # 4. MOMENTOS CRÍTICOS
            # Picos de intensidade (valor absoluto > 30)
            picos = np.sum(np.abs(valores) > 30)
            metrics['picos_intensidade'] = picos
            
            # Sequências de dominância
            # Sequências positivas consecutivas (> 10)
            sequencias_pos = []
            sequencias_neg = []
            current_seq = 0
            current_val = 0
            
            for v in valores:
                if v > 10:
                    if current_val <= 0:
                        if current_val < 0 and current_seq > 0:
                            sequencias_neg.append(current_seq)
                        current_seq = 1
                    else:
                        current_seq += 1
                    current_val = 1
                elif v < -10:
                    if current_val >= 0:
                        if current_val > 0 and current_seq > 0:
                            sequencias_pos.append(current_seq)
                        current_seq = 1
                    else:
                        current_seq += 1
                    current_val = -1
                else:
                    if current_val > 0 and current_seq > 0:
                        sequencias_pos.append(current_seq)
                    elif current_val < 0 and current_seq > 0:
                        sequencias_neg.append(current_seq)
                    current_seq = 0
                    current_val = 0
            
            metrics['maior_sequencia_pos'] = max(sequencias_pos) if sequencias_pos else 0
            metrics['maior_sequencia_neg'] = max(sequencias_neg) if sequencias_neg else 0
            metrics['total_sequencias_pos'] = len(sequencias_pos)
            metrics['total_sequencias_neg'] = len(sequencias_neg)
            
            # 5. DINÂMICA TEMPORAL
            # Tendência linear (usando regressão)
            if len(valores) > 1:
                x = np.arange(len(valores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, valores)
                metrics['tendencia_linear'] = slope * len(valores)  # Mudança total ao longo do jogo
                metrics['r2_tendencia'] = r_value ** 2
            else:
                metrics['tendencia_linear'] = 0
                metrics['r2_tendencia'] = 0
            
            # 6. MÉTRICAS DE EQUILÍBRIO
            valores_pos = valores[valores > 0]
            valores_neg = valores[valores < 0]
            
            metrics['dominancia_pos'] = np.sum(valores_pos) / (np.sum(np.abs(valores)) + 1e-10)
            metrics['dominancia_neg'] = np.sum(np.abs(valores_neg)) / (np.sum(np.abs(valores)) + 1e-10)
            
            # 7. VOLATILIDADE (janelas de 10 minutos)
            volatilidades = []
            window_size = 10
            for i in range(0, len(valores) - window_size + 1, window_size):
                window = valores[i:i+window_size]
                if len(window) > 1:
                    volatilidades.append(np.std(window))
            
            metrics['volatilidade_media'] = np.mean(volatilidades) if volatilidades else 0
            metrics['volatilidade_max'] = np.max(volatilidades) if volatilidades else 0
            
            # 8. MÉTRICAS DE AÇÃO
            # Taxa de mudança absoluta
            if len(valores) > 1:
                mudancas = np.abs(np.diff(valores))
                metrics['taxa_mudanca_media'] = np.mean(mudancas)
                metrics['taxa_mudanca_max'] = np.max(mudancas)
            else:
                metrics['taxa_mudanca_media'] = 0
                metrics['taxa_mudanca_max'] = 0
            
            return metrics

        def encontrar_json_graph(id_jogo):
            """Encontra o arquivo JSON graph no formato: {id_jogo}/graph.json"""
            id_str = str(id_jogo).strip()
            
            # Padrão: /playwright_jsons_full/{id_jogo}/graph.json
            json_path = JSONS_DIR / id_str / "graph.json"
            
            if json_path.exists():
                return json_path
            
            # Tentar variações de nome do arquivo
            variações = [
                JSONS_DIR / id_str / f"{id_str}_graph.json",
                JSONS_DIR / id_str / "graph_data.json",
                JSONS_DIR / f"event_{id_str}" / "graph.json",
                JSONS_DIR / f"{id_str}" / "points.json",
            ]
            
            for var in variações:
                if var.exists():
                    return var
            
            return None

        def obter_pontos_jogo_local(id_jogo):
            """
            Obtém dados de pontos do JSON local no formato: {id_jogo}/graph.json
            Retorna DataFrame com minutos e métricas adicionais
            """
            try:
                # Encontrar arquivo JSON
                json_path = encontrar_json_graph(id_jogo)
                
                if json_path is None or not json_path.exists():
                    if verbose:
                        print(f"JSON não encontrado para jogo {id_jogo}")
                    return None, None
                
                if verbose:
                    print(f"Lendo JSON: {json_path}")
                
                # Ler arquivo JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Verificar estrutura do JSON
                if not isinstance(json_data, dict):
                    if verbose:
                        print(f"JSON inválido para jogo {id_jogo}: não é um dicionário")
                    return None, None
                
                # Extrair dados do graph - verificar estrutura fornecida
                if 'data' in json_data and 'graphPoints' in json_data['data']:
                    graph_data = json_data['data']['graphPoints']
                elif 'graph' in json_data:
                    graph_data = json_data['graph']
                elif 'graphPoints' in json_data:
                    graph_data = json_data['graphPoints']
                else:
                    if verbose:
                        print(f"Estrutura de dados não reconhecida para jogo {id_jogo}")
                    return None, None
                
                if not graph_data:
                    if verbose:
                        print(f"JSON sem dados de gráfico para jogo {id_jogo}")
                    return None, None
                
                # Coletar minutos e valores
                minutes = []
                values = []
                
                for item in graph_data:
                    if isinstance(item, dict):
                        minute = item.get('minute')
                        value = item.get('value')
                        
                        # Filtrar minutos fracionários (ex: 45.5, 90.5)
                        if minute is not None and value is not None:
                            # Considerar apenas minutos inteiros de 1 a 90
                            if isinstance(minute, (int, float)):
                                min_int = int(minute)
                                if 1 <= min_int <= minute_limit:
                                    minutes.append(min_int)
                                    values.append(float(value))
                
                if not minutes or not values:
                    if verbose:
                        print(f"Sem dados de minutos/valores válidos para jogo {id_jogo}")
                    return None, None
                
                # Criar DataFrame de minutos (1 a 90)
                df_minutes = pd.DataFrame(index=[0], columns=minute_cols_str).fillna(0.0)
                
                # Preencher valores
                for min_int, val in zip(minutes, values):
                    col_name = str(min_int)
                    if col_name in df_minutes.columns:
                        df_minutes.at[0, col_name] = float(val)
                
                # Calcular métricas de ritmo
                metrics = calcular_metricas_ritmo(df_minutes[minute_cols_str].astype(float))
                
                return df_minutes, metrics

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Erro ao decodificar JSON para jogo {id_jogo}: {e}")
                return None, None
            except KeyError as e:
                if verbose:
                    print(f"Chave não encontrada no JSON para jogo {id_jogo}: {e}")
                return None, None
            except Exception as e:
                if verbose:
                    print(f"Erro ao processar JSON local para jogo {id_jogo}: {e}")
                return None, None

        # --- Carregar arquivo com IDs ---
        ids_path = TODAY_FOLDER / ids_source_file
        if not ids_path.exists():
            ids_path = BASE / ids_source_file
        if not ids_path.exists():
            raise FileNotFoundError(f"Arquivo com IDs não encontrado: {ids_source_file}")

        df_ids = pd.read_csv(ids_path, low_memory=False, dtype=str)
        df_ids = _normalize_id_col(df_ids, 'ID_Jogo')

        if 'ID_Jogo' not in df_ids.columns:
            raise KeyError("Coluna 'ID_Jogo' não encontrada no arquivo de IDs")

        # --- Verificar se a pasta de JSONs existe ---
        if not JSONS_DIR.exists():
            print(f"AVISO: Pasta de JSONs não encontrada: {JSONS_DIR}")
            print(f"Criando pasta: {JSONS_DIR}")
            JSONS_DIR.mkdir(parents=True, exist_ok=True)
            return pd.DataFrame()

        # --- Obter todos os IDs únicos ---
        all_ids = df_ids['ID_Jogo'].dropna().unique().tolist()
        
        if verbose:
            print(f"Total de IDs encontrados: {len(all_ids)}")
            print(f"Pasta de JSONs: {JSONS_DIR}")
            print(f"Procurando arquivos no formato: {JSONS_DIR}/{{id_jogo}}/graph.json")
        
        # --- Contar quantos JSONs existem ---
        json_count = 0
        for id_jogo in all_ids:
            id_str = str(id_jogo).strip()
            json_path = JSONS_DIR / id_str / "graph.json"
            if json_path.exists():
                json_count += 1
        
        if verbose:
            print(f"JSONs encontrados: {json_count}/{len(all_ids)}")
        
        # --- PROCESSAR TODOS OS JOGOS ---
        resultados = []
        sem_dados = []
        com_dados = []
        
        for i, id_jogo in enumerate(all_ids, start=1):
            if verbose:
                print(f"[{i}/{len(all_ids)}] Processando {id_jogo}")
            
            try:
                df_pontos, metrics = obter_pontos_jogo_local(id_jogo)
                
                if df_pontos is None or metrics is None:
                    sem_dados.append(id_jogo)
                    # Criar entrada vazia
                    df_pontos = pd.DataFrame({str(m): 0.0 for m in range(1, minute_limit+1)}, index=[0])
                    metrics = calcular_metricas_ritmo(df_pontos[minute_cols_str].astype(float))
                else:
                    com_dados.append(id_jogo)
                
                # Garantir todas as colunas de minutos
                for m in minute_cols_str:
                    if m not in df_pontos.columns:
                        df_pontos[m] = 0.0

                df_pontos = df_pontos.reset_index(drop=True)
                
                # Adicionar ID e métricas ao DataFrame
                df_pontos['ID_Jogo'] = str(id_jogo)
                
                # Adicionar todas as métricas como colunas
                for metric_name, metric_value in metrics.items():
                    df_pontos[metric_name] = metric_value
                
                resultados.append(df_pontos)

            except Exception as e:
                if verbose:
                    print(f"Erro ao processar jogo {id_jogo}: {e}")
                # Criar entrada vazia em caso de erro
                df_pontos = pd.DataFrame({str(m): 0.0 for m in range(1, minute_limit+1)}, index=[0])
                metrics = calcular_metricas_ritmo(df_pontos[minute_cols_str].astype(float))
                df_pontos['ID_Jogo'] = str(id_jogo)
                for metric_name, metric_value in metrics.items():
                    df_pontos[metric_name] = metric_value
                resultados.append(df_pontos)
                sem_dados.append(id_jogo)
        
        # --- CONSTRUIR DATAFRAME FINAL ---
        if not resultados:
            print("Nenhum resultado obtido")
            return pd.DataFrame()
        
        df_final = pd.concat(resultados, ignore_index=True)
        df_final = _normalize_id_col(df_final, 'ID_Jogo')

        # Garantir colunas de minutos
        for m in minute_cols_str:
            if m not in df_final.columns:
                df_final[m] = 0.0
            else:
                df_final[m] = pd.to_numeric(df_final[m], errors='coerce').fillna(0).astype(float)

        # Calcular médias básicas
        try:
            df_final['media_pontos_casa'] = df_final[minute_cols_str].where(df_final[minute_cols_str] > 0).mean(axis=1, skipna=True).fillna(1)
            df_final['media_pontos_fora'] = df_final[minute_cols_str].where(df_final[minute_cols_str] < 0).abs().mean(axis=1, skipna=True).fillna(1)
        except:
            df_final['media_pontos_casa'] = 1.0
            df_final['media_pontos_fora'] = 1.0

        # Ordem final das colunas
        # Primeiro as colunas básicas, depois métricas de ritmo
        colunas_basicas = ['ID_Jogo'] + minute_cols_str + ['media_pontos_casa', 'media_pontos_fora']
        
        # Todas as outras colunas (métricas de ritmo)
        outras_colunas = [col for col in df_final.columns if col not in colunas_basicas and col != 'ID_Jogo']
        
        # Reordenar
        ordem_colunas = colunas_basicas + sorted(outras_colunas)
        df_final = df_final[ordem_colunas]
        
        # Remover duplicados
        df_final = df_final.drop_duplicates(subset=['ID_Jogo'], keep='first').reset_index(drop=True)

        # --- SALVAR ---
        if save:
            out_path = TODAY_FOLDER / out_filename
            df_final.to_csv(out_path, index=False, encoding='utf-8-sig')
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESUMO DA COLETA COM MÉTRICAS DE RITMO:")
                print(f"{'='*80}")
                print(f"Total de jogos processados: {len(df_final)}")
                print(f"Jogos com dados de gráfico: {len(com_dados)}")
                print(f"Jogos sem dados (preenchidos com zeros): {len(sem_dados)}")
                print(f"Colunas no CSV final: {len(df_final.columns)}")
                print(f"CSV gravado em: {out_path}")
                print(f"{'='*80}")
                
                if sem_dados and verbose:
                    print(f"\nJogos sem dados de gráfico (primeiros 10):")
                    for jogo in sem_dados[:10]:
                        print(f"  - {jogo}")
                    if len(sem_dados) > 10:
                        print(f"  ... e mais {len(sem_dados) - 10} jogos")
                
                # Mostrar amostra com algumas métricas importantes
                print("\nAmostra dos dados (primeiras 3 linhas - métricas selecionadas):")
                colunas_mostrar = ['ID_Jogo', 'intensidade_total', 'intensidade_media', 'variabilidade', 
                                  'picos_intensidade', 'tendencia_linear', 'volatilidade_media']
                colunas_disponiveis = [col for col in colunas_mostrar if col in df_final.columns]
                
                if colunas_disponiveis:
                    print(df_final[colunas_disponiveis].head(3).to_string())
                
                # Estatísticas das métricas
                print("\nEstatísticas das principais métricas:")
                metricas_importantes = ['intensidade_total', 'intensidade_media', 'variabilidade', 
                                       'picos_intensidade', 'dominancia_pos', 'volatilidade_media']
                
                for metrica in metricas_importantes:
                    if metrica in df_final.columns:
                        valores = df_final[metrica]
                        print(f"{metrica:20s}: Média={valores.mean():.2f}, Min={valores.min():.2f}, Max={valores.max():.2f}")

        return df_final

    except Exception as e:
        print(f"Erro em Coleta_pontos_momentum_completa: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Função de compatibilidade (mantém interface original)
def Coleta_pontos_momentum(
    ids_source_file: str = 'df_previsoes_sim_concatenado.csv',
    out_filename: str = 'dados_pontos_geral.csv',
    minute_limit: int = 90,
    save: bool = True,
    verbose: bool = True
):
    """Wrapper para manter compatibilidade com código existente"""
    # Chama a função completa
    df = Coleta_pontos_momentum_completa(
        ids_source_file=ids_source_file,
        out_filename=out_filename,
        minute_limit=minute_limit,
        save=save,
        verbose=verbose
    )
    return df

# Para executar diretamente
if __name__ == "__main__":
    print("Iniciando coleta de pontos com métricas de ritmo...")
    df_result = Coleta_pontos_momentum(verbose=True)
    
    if df_result.empty:
        print("Nenhum dado coletado.")
    else:
        print(f"\nColeta concluída. {len(df_result)} jogos processados.")
        print(f"Arquivo salvo em: data/dados_pontos_geral.csv")
        print(f"Métricas disponíveis: {len(df_result.columns) - 92} métricas adicionais")