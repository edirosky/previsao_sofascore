# @title CARREGAMENTO DE MODELOS - VERSÃO COMPLETA (SEM FILTROS)
def Carregar_modelos_completo():
    """
    Versão COMPLETA do carregamento de modelos que:
    1. Processa TODOS os jogos do CSV incidentes_estatisticas_geral.csv
    2. Gera previsões para TODOS os jogos sem filtros
    3. Adiciona sufixo _depois_previsao nas colunas geradas
    4. Atualiza o CSV mestre com todas as previsões
    """
    try:
        import os, json, joblib, pandas as pd, numpy as np, warnings, shutil, traceback
        from datetime import datetime, timezone
        import pytz
        from pathlib import Path

        # Suprimir warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        print("=" * 70)
        print("🚀 CARREGAMENTO DE MODELOS - VERSÃO COMPLETA (SEM FILTROS)")
        print("   Processando TODOS os jogos do CSV incidentes_estatisticas_geral.csv")
        print("=" * 70)

        # ============================
        # FUNÇÕES AUXILIARES
        # ============================
        
        def load_csv_file(file_path: str) -> pd.DataFrame:
            """Carrega arquivo CSV"""
            try:
                df = pd.read_csv(file_path, dtype=str, encoding='utf-8-sig')
                print(f"✅ CSV carregado: {file_path}")
                print(f"📊 Shape: {df.shape}")
                print(f"📋 Colunas: {len(df.columns)}")
                print(f"📈 Linhas: {len(df)}")
                return df
            except Exception as e:
                print(f"❌ Erro ao carregar o arquivo CSV: {e}")
                return pd.DataFrame()

        def convert_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
            """Converte colunas para numérico"""
            df = df.copy()
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.'), 
                            errors='coerce'
                        ).fillna(0)
                    except Exception:
                        df[col] = 0
                else:
                    df[col] = 0
            return df

        def parse_score(score_str):
            """Parseia string de placar no formato 'X-Y'"""
            try:
                if pd.isna(score_str) or score_str == '' or str(score_str).strip() == 'nan':
                    return 0, 0
                score_str = str(score_str).strip()
                if '-' in score_str:
                    parts = score_str.split('-')
                    home = int(float(parts[0].strip())) if parts[0].strip() != '' else 0
                    away = int(float(parts[1].strip())) if parts[1].strip() != '' else 0
                    return home, away
                else:
                    return 0, 0
            except Exception:
                return 0, 0

        def prepare_placars(df: pd.DataFrame) -> pd.DataFrame:
            """Prepara os placares HT e FT"""
            df = df.copy()
            
            # HT - verificar múltiplas possibilidades de nomenclatura
            ht_found = False
            for ht_col in ['PLACAR_HT', 'PLACAR HT', 'PLACAR_HT_depois_previsao', 'PLACAR HT_depois_previsao']:
                if ht_col in df.columns:
                    ht_scores = df[ht_col].apply(parse_score)
                    df['HT_Casa'] = [score[0] for score in ht_scores]
                    df['HT_Fora'] = [score[1] for score in ht_scores]
                    ht_found = True
                    break
            
            if not ht_found:
                df['HT_Casa'] = 0
                df['HT_Fora'] = 0
            
            # FT - verificar múltiplas possibilidades
            ft_found = False
            for ft_col in ['PLACAR_FT', 'PLACAR FT', 'PLACAR_FT_depois_previsao', 'PLACAR FT_depois_previsao']:
                if ft_col in df.columns:
                    ft_scores = df[ft_col].apply(parse_score)
                    df['FT_Casa'] = [score[0] for score in ft_scores]
                    df['FT_Fora'] = [score[1] for score in ft_scores]
                    ft_found = True
                    break
            
            if not ft_found:
                # Tentar usar totais de golos
                if 'total_golos_casa_depois_previsao' in df.columns and 'total_golos_fora_depois_previsao' in df.columns:
                    df['FT_Casa'] = pd.to_numeric(df['total_golos_casa_depois_previsao'], errors='coerce').fillna(0)
                    df['FT_Fora'] = pd.to_numeric(df['total_golos_fora_depois_previsao'], errors='coerce').fillna(0)
                elif 'total_golos_casa' in df.columns and 'total_golos_fora' in df.columns:
                    df['FT_Casa'] = pd.to_numeric(df['total_golos_casa'], errors='coerce').fillna(0)
                    df['FT_Fora'] = pd.to_numeric(df['total_golos_fora'], errors='coerce').fillna(0)
                else:
                    df['FT_Casa'] = df['HT_Casa']
                    df['FT_Fora'] = df['HT_Fora']
            
            return df

        def calcular_metricas_avancadas(df: pd.DataFrame) -> pd.DataFrame:
            """Calcula métricas avançadas para features dos modelos"""
            df = df.copy()
            
            # Garantir que colunas numéricas existem
            numeric_cols = [
                'Ball_possession_Casa', 'Ball_possession_Fora',
                'Accurate_passes_Casa', 'Passes_Casa',
                'Accurate_passes_Fora', 'Passes_Fora',
                'Total_shots_Casa', 'Total_shots_Fora',
                'Shots_inside_box_Casa', 'Shots_on_target_Casa',
                'Shots_inside_box_Fora', 'Shots_on_target_Fora',
                'Final_third_entries_Casa', 'Final_third_entries_Fora',
                'Tackles_Casa', 'Tackles_Fora',
                'Corner_kicks_Casa', 'Corner_kicks_Fora',
                'HT_Casa', 'HT_Fora', 'FT_Casa', 'FT_Fora'
            ]
            
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Possession ratio
            bp_casa = df['Ball_possession_Casa']
            bp_fora = df['Ball_possession_Fora']
            total_possession = bp_casa + bp_fora + 1e-6
            df['Possession_ratio_home'] = bp_casa / total_possession
            df['Possession_ratio_away'] = bp_fora / total_possession

            # Pass accuracy
            acc_passes_casa = df['Accurate_passes_Casa']
            passes_casa = df['Passes_Casa']
            acc_passes_fora = df['Accurate_passes_Fora']
            passes_fora = df['Passes_Fora']

            df['Pass_accuracy_home'] = np.where(
                passes_casa > 0,
                100 * acc_passes_casa / passes_casa,
                0
            )
            df['Pass_accuracy_away'] = np.where(
                passes_fora > 0,
                100 * acc_passes_fora / passes_fora,
                0
            )

            # Shot conversion
            ht_casa = df['HT_Casa']
            ht_fora = df['HT_Fora']
            total_shots_casa = df['Total_shots_Casa']
            total_shots_fora = df['Total_shots_Fora']
            
            df['Shot_conversion_home'] = np.where(
                total_shots_casa > 0,
                ht_casa / total_shots_casa,
                0
            )
            df['Shot_conversion_away'] = np.where(
                total_shots_fora > 0,
                ht_fora / total_shots_fora,
                0
            )

            # xG proxy
            shots_inside_box_casa = df['Shots_inside_box_Casa']
            shots_on_target_casa = df['Shots_on_target_Casa']
            shots_inside_box_fora = df['Shots_inside_box_Fora']
            shots_on_target_fora = df['Shots_on_target_Fora']
            
            df['xG_proxy_home'] = (shots_inside_box_casa * 0.3) + (shots_on_target_casa * 0.2)
            df['xG_proxy_away'] = (shots_inside_box_fora * 0.3) + (shots_on_target_fora * 0.2)

            # Pressure index
            final_third_entries_casa = df['Final_third_entries_Casa']
            tackles_fora = df['Tackles_Fora']
            final_third_entries_fora = df['Final_third_entries_Fora']
            tackles_casa = df['Tackles_Casa']
            
            df['Pressure_index_home'] = np.where(
                tackles_fora > 0,
                final_third_entries_casa / tackles_fora,
                0
            )
            df['Pressure_index_away'] = np.where(
                tackles_casa > 0,
                final_third_entries_fora / tackles_casa,
                0
            )

            # Attacking index
            corner_kicks_casa = df['Corner_kicks_Casa']
            corner_kicks_fora = df['Corner_kicks_Fora']
            
            df['Attacking_index_home'] = (
                0.4 * shots_on_target_casa +
                0.3 * shots_inside_box_casa +
                0.2 * final_third_entries_casa +
                0.1 * corner_kicks_casa
            )
            df['Attacking_index_away'] = (
                0.4 * shots_on_target_fora +
                0.3 * shots_inside_box_fora +
                0.2 * final_third_entries_fora +
                0.1 * corner_kicks_fora
            )

            # Garantir que não há NaN
            new_columns = [
                'Possession_ratio_home', 'Possession_ratio_away', 'Pass_accuracy_home',
                'Pass_accuracy_away', 'Shot_conversion_home', 'Shot_conversion_away',
                'xG_proxy_home', 'xG_proxy_away', 'Pressure_index_home', 'Pressure_index_away',
                'Attacking_index_home', 'Attacking_index_away'
            ]

            for col in new_columns:
                df[col] = df[col].fillna(0)

            return df

        def prepare_features_and_targets(df: pd.DataFrame):
            """Prepara features e targets para os modelos"""
            df = prepare_placars(df)
            df = calcular_metricas_avancadas(df)
            
            features = [
                'Possession_ratio_home', 'Possession_ratio_away',
                'Pass_accuracy_home', 'Pass_accuracy_away',
                'Shot_conversion_home', 'Shot_conversion_away',
                'xG_proxy_home', 'xG_proxy_away',
                'Pressure_index_home', 'Pressure_index_away',
                'Attacking_index_home', 'Attacking_index_away'
            ]
            
            for feat in features:
                if feat not in df.columns:
                    df[feat] = 0
            
            X = df[features].fillna(0)
            X.columns = X.columns.astype(str)
            
            # Criar targets
            df['Mais_0.5_Golos_SegundaParte'] = (
                (df['FT_Casa'] - df['HT_Casa'] + df['FT_Fora'] - df['HT_Fora']) > 0.5
            ).astype(int)
            
            df['Mais_1.5_Golos_SegundaParte'] = (
                (df['FT_Casa'] - df['HT_Casa'] + df['FT_Fora'] - df['HT_Fora']) > 1.5
            ).astype(int)
            
            df['Equipa_Perdendo_Marcar_SegundaParte'] = (
                ((df['HT_Casa'] < df['HT_Fora']) & (df['FT_Casa'] - df['HT_Casa'] >= 1)) |
                ((df['HT_Fora'] < df['HT_Casa']) & (df['FT_Fora'] - df['HT_Fora'] >= 1))
            ).astype(int)
            
            targets = [
                'Mais_0.5_Golos_SegundaParte',
                'Mais_1.5_Golos_SegundaParte',
                'Equipa_Perdendo_Marcar_SegundaParte'
            ]
            
            return X, df, targets

        def load_models_and_predict(X, targets, model_dir: str):
            """Carrega modelos e faz previsões"""
            predictions = {}
            X_clean = X.copy()
            X_clean.columns = [str(col).strip() for col in X_clean.columns]
            
            for target in targets:
                model_path = os.path.join(model_dir, f"{target}_VotingEnsemble.pkl")
                
                if not os.path.exists(model_path):
                    print(f"⚠️ Modelo {target} não encontrado em {model_path}")
                    predictions[f"pred_{target}"] = np.zeros(len(X_clean), dtype=int)
                    predictions[f"pred_{target}_proba"] = np.zeros(len(X_clean))
                    continue
                
                try:
                    model = joblib.load(model_path)
                    print(f"✅ Modelo {target} carregado")
                    
                    expected_features = None
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = [str(feat).strip() for feat in model.feature_names_in_]
                    else:
                        for estimator in getattr(model, 'estimators_', []):
                            if hasattr(estimator, 'feature_names_in_'):
                                expected_features = [str(feat).strip() for feat in estimator.feature_names_in_]
                                break
                    
                    X_final = X_clean.copy()
                    
                    if expected_features:
                        missing_features = set(expected_features) - set(X_final.columns)
                        extra_features = set(X_final.columns) - set(expected_features)
                        
                        for feature in missing_features:
                            X_final[feature] = 0
                        
                        if extra_features:
                            X_final = X_final.drop(columns=list(extra_features))
                        
                        try:
                            X_final = X_final[expected_features]
                        except KeyError:
                            pass
                    
                    X_final = X_final.fillna(0).astype(float)
                    predictions[f"pred_{target}"] = model.predict(X_final)
                    
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X_final)[:, 1]
                            predictions[f"pred_{target}_proba"] = proba
                        except Exception as e:
                            print(f"⚠️ Erro ao calcular probabilidades para {target}: {e}")
                            predictions[f"pred_{target}_proba"] = np.zeros(len(X_final))
                    else:
                        predictions[f"pred_{target}_proba"] = np.zeros(len(X_final))
                    
                except Exception as e:
                    print(f"❌ Erro ao carregar/prever com modelo {target}: {e}")
                    predictions[f"pred_{target}"] = np.zeros(len(X_clean), dtype=int)
                    predictions[f"pred_{target}_proba"] = np.zeros(len(X_clean))
            
            return predictions

        def create_final_dataframe(df, predictions):
            """Cria DataFrame final com previsões"""
            pred_df = pd.DataFrame(predictions, index=df.index)
            
            # Colunas base importantes
            base_columns = [
                "ID_Jogo", "Torneio", "Temporada", "Time_Home", "Time_Away",
                "Status", "Tipo_Status", "Inicio", "Atual_time", 
                "Minutos_jogo", "evolução do Placar",
                "total_golos_casa", "total_golos_fora", "Baixar"
            ]
            
            base_data = {}
            for col in base_columns:
                if col in df.columns:
                    base_data[col] = df[col]
                else:
                    # Tentar encontrar coluna com sufixo _depois_previsao
                    col_with_suffix = f"{col}_depois_previsao"
                    if col_with_suffix in df.columns:
                        base_data[col] = df[col_with_suffix]
                    else:
                        if col == "ID_Jogo":
                            base_data[col] = df.index.astype(str)
                        elif col in ["total_golos_casa", "total_golos_fora"]:
                            base_data[col] = 0
                        else:
                            base_data[col] = ""
            
            base_df = pd.DataFrame(base_data)
            df_final = pd.concat([base_df, pred_df.copy()], axis=1, ignore_index=False)
            
            # Formatar probabilidades
            prediction_proba_columns = [col for col in pred_df.columns if "_proba" in col]
            prediction_binary_columns = [col for col in pred_df.columns if "_proba" not in col]
            
            for col in prediction_proba_columns:
                if col in df_final.columns:
                    df_final[col] = (pd.to_numeric(df_final[col], errors='coerce').fillna(0) * 100).round(2)
                    df_final[col] = df_final[col].astype(str) + "%"
            
            concept_mapping = {1: "Sim", 0: "Não"}
            for col in prediction_binary_columns:
                if col in df_final.columns:
                    conceito_col = col.replace("pred_", "conceito_")
                    df_final[conceito_col] = df_final[col].map(concept_mapping).fillna("Não")
            
            return df_final

        def save_dataframe(df, output_path: str):
            """Salva DataFrame em CSV"""
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"💾 DataFrame salvo em: {output_path}")
            print(f"   Shape: {df.shape}")

        def concatenate_sim_predictions(df, conceito_cols: list) -> pd.DataFrame:
            """Concatena previsões 'Sim' em uma única coluna"""
            def concatenar_sim(row):
                sim_list = []
                for col in conceito_cols:
                    if col in row and row.get(col) == "Sim":
                        target_name = col.replace("conceito_", "")
                        proba_col = "pred_" + target_name + "_proba"
                        proba_val = row.get(proba_col, "0%")
                        sim_list.append(f"{target_name} ({proba_val})")
                return "; ".join(sim_list) if sim_list else ""
            
            if all(col in df.columns for col in conceito_cols):
                df["Previsao_Sim_Concatenado"] = df.apply(concatenar_sim, axis=1)
            else:
                df["Previsao_Sim_Concatenado"] = ""
            
            return df

        def criar_metricas_combinadas_ajustadas(df):
            """Cria métricas combinadas com thresholds ajustados"""
            print("🎯 Criando métricas combinadas com confiança ajustada...")

            proba_columns = [
                'pred_Mais_0.5_Golos_SegundaParte_proba',
                'pred_Mais_1.5_Golos_SegundaParte_proba',
                'pred_Equipa_Perdendo_Marcar_SegundaParte_proba'
            ]

            for col in proba_columns:
                if col not in df.columns:
                    df[col] = "0%"

            df['proba_0.5_num'] = df['pred_Mais_0.5_Golos_SegundaParte_proba'].str.replace('%', '').astype(float).fillna(0)
            df['proba_1.5_num'] = df['pred_Mais_1.5_Golos_SegundaParte_proba'].str.replace('%', '').astype(float).fillna(0)
            df['proba_equipa_num'] = df['pred_Equipa_Perdendo_Marcar_SegundaParte_proba'].str.replace('%', '').astype(float).fillna(0)

            # MÉTRICA 1: Média Ponderada
            df['media_ponderada_3_conceitos'] = (
                (df['proba_0.5_num'] * 4) +
                (df['proba_1.5_num'] * 2) +
                (df['proba_equipa_num'] * 1)
            ) / 7
            df['previsao_media_ponderada'] = (df['media_ponderada_3_conceitos'] >= 35).map({True: 'Sim', False: 'Não'})

            # MÉTRICA 2: Média Simples
            df['media_2_conceitos'] = (df['proba_0.5_num'] + df['proba_1.5_num']) / 2
            df['previsao_media_simples'] = (df['media_2_conceitos'] >= 40).map({True: 'Sim', False: 'Não'})

            # MÉTRICA 3: Sistema de Pontuação
            def calcular_pontuacao_ajustada(row):
                pontuacao = 0
                if row['proba_0.5_num'] >= 45:
                    pontuacao += 3
                elif row['proba_0.5_num'] >= 30:
                    pontuacao += 2
                elif row['proba_0.5_num'] >= 20:
                    pontuacao += 1

                if row['proba_1.5_num'] >= 40:
                    pontuacao += 2
                elif row['proba_1.5_num'] >= 25:
                    pontuacao += 1

                if row['proba_equipa_num'] >= 50:
                    pontuacao += 1

                return pontuacao

            df['pontuacao_ajustada'] = df.apply(calcular_pontuacao_ajustada, axis=1)
            df['previsao_pontuacao_ajustada'] = (df['pontuacao_ajustada'] >= 3).map({True: 'Sim', False: 'Não'})

            # MÉTRICA 4: Probabilidade Máxima
            df['max_probabilidade'] = df[['proba_0.5_num', 'proba_1.5_num']].max(axis=1)
            
            def threshold_inteligente_ajustado(max_prob, prob_0_5):
                if max_prob == prob_0_5 and max_prob >= 35:
                    return 'Sim'
                elif max_prob >= 45:
                    return 'Sim'
                else:
                    return 'Não'

            df['previsao_max_inteligente'] = df.apply(
                lambda row: threshold_inteligente_ajustado(row['max_probabilidade'], row['proba_0.5_num']), axis=1
            )

            # MÉTRICA 5: Ensemble Híbrido
            def ensemble_hibrido_ajustado(row):
                criterios = 0
                if row['proba_0.5_num'] >= 35:
                    criterios += 1
                if row['proba_1.5_num'] >= 40:
                    criterios += 1
                if row['media_ponderada_3_conceitos'] >= 30:
                    criterios += 1
                return 'Sim' if criterios >= 2 else 'Não'

            df['previsao_ensemble_hibrido'] = df.apply(ensemble_hibrido_ajustado, axis=1)

            # PREVISÃO CONSENSUAL
            metricas_ajustadas = [
                'previsao_media_ponderada',
                'previsao_media_simples',
                'previsao_pontuacao_ajustada',
                'previsao_max_inteligente',
                'previsao_ensemble_hibrido'
            ]

            for col in metricas_ajustadas:
                if col not in df.columns:
                    df[col] = 'Não'

            df['concordancia_ajustada'] = df[metricas_ajustadas].apply(lambda x: (x == 'Sim').sum(), axis=1)
            df['previsao_consensual_ajustada'] = (df['concordancia_ajustada'] >= 2).map({True: 'Sim', False: 'Não'})

            # SCORE DE CONFIANÇA
            df['score_confianca_ajustado'] = (
                df['proba_0.5_num'] * 0.5 +
                df['proba_1.5_num'] * 0.3 +
                df['media_ponderada_3_conceitos'] * 0.2
            ).fillna(0)

            # CLASSIFICAÇÃO DE CONFIANÇA
            def classificar_confianca_ajustada(score):
                if score >= 75:
                    return "Alta Confiança"
                elif score >= 65:
                    return "Média Confiança"
                elif score >= 50:
                    return "Baixa Confiança"
                else:
                    return "Muito Baixa Confiança"

            df['nivel_confianca_ajustado'] = df['score_confianca_ajustado'].apply(classificar_confianca_ajustada)

            # Limpar colunas auxiliares
            colunas_manter = [col for col in df.columns if not col.endswith('_num')]
            return df[colunas_manter]

        def determine_tipo_previsao(game_id, existing_df):
            """Determina se é uma nova previsão ou atualização"""
            if not existing_df.empty and str(game_id) in existing_df['ID_Jogo'].astype(str).values:
                return "ATUALIZAÇÃO PREVISÃO"
            else:
                return "NOVA PREVISÃO"

        def dedupe_keep_latest_only(csv_path: str):
            """Remove duplicatas mantendo a última versão de cada jogo"""
            if not os.path.exists(csv_path):
                return

            df_all = pd.read_csv(csv_path, dtype=str).fillna(0)
            
            if 'Timestamp' in df_all.columns:
                df_all['Timestamp_dt'] = pd.to_datetime(df_all['Timestamp'], errors='coerce')
                df_all['__order_idx'] = range(len(df_all))
                df_all['Timestamp_sort'] = df_all['Timestamp_dt'].fillna(
                    pd.to_datetime(df_all['__order_idx'], unit='s', errors='coerce')
                )
                sort_col = 'Timestamp_sort'
            else:
                df_all['__order_idx'] = range(len(df_all))
                sort_col = '__order_idx'

            df_sorted = df_all.sort_values(by=sort_col, ascending=False)
            df_keep = df_sorted.drop_duplicates(subset=['ID_Jogo'], keep='first')
            df_keep = df_keep.sort_values(by=sort_col, ascending=True)

            # Usar datetime.now com timezone UTC
            bak_path = csv_path.replace('.csv', f'.dedupe.bak.{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.csv')
            try:
                shutil.copy2(csv_path, bak_path)
            except Exception:
                pass

            df_keep.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ Deduplicação concluída. Backup salvo: {os.path.basename(bak_path)}")
            print(f"📊 Antes: {len(df_all)} linhas | Depois: {len(df_keep)} linhas")

            if 'Tipo_Previsao' in df_keep.columns:
                tipos_count = df_keep['Tipo_Previsao'].value_counts()
                print(f"\n📈 DISTRIBUIÇÃO DOS TIPOS DE PREVISÃO:")
                for tipo, count in tipos_count.items():
                    print(f"   • {tipo}: {count} jogos")

        # ============================
        # EXECUÇÃO PRINCIPAL
        # ============================
        
        BASE_DIR = "/workspaces/previsao_sofascore"
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MODELS_DIR = os.path.join(BASE_DIR, "models")
        
        input_file = os.path.join(DATA_DIR, "incidentes_estatisticas_geral.csv")
        output_file = os.path.join(DATA_DIR, "df_previsoes_live.csv")
        output_concatenado = os.path.join(DATA_DIR, "df_previsoes_sim_concatenado.csv")
        
        # 1. Carregar CSV
        print("\n1️⃣ Carregando dados do CSV...")
        df = load_csv_file(input_file)
        
        if df.empty:
            print("❌ DataFrame vazio. Verifique se o arquivo existe:")
            print(f"   Caminho: {input_file}")
            return
        
        # 2. Limpar dados
        print("\n2️⃣ Limpando dados...")
        df = df.replace(['', 'nan', 'NaN', 'None', 'null'], np.nan)
        df = df.fillna(0)
        
        # 3. Converter colunas numéricas
        print("\n3️⃣ Convertendo colunas numéricas...")
        numeric_cols = [
            "Ball_possession_Casa", "Ball_possession_Fora",
            "Accurate_passes_Casa", "Passes_Casa",
            "Accurate_passes_Fora", "Passes_Fora",
            "Total_shots_Casa", "Total_shots_Fora",
            "Shots_inside_box_Casa", "Shots_on_target_Casa",
            "Shots_inside_box_Fora", "Shots_on_target_Fora",
            "Final_third_entries_Casa", "Final_third_entries_Fora",
            "Tackles_Casa", "Tackles_Fora",
            "Corner_kicks_Casa", "Corner_kicks_Fora",
            "Placar_Home", "Placar_Away",
            "total_golos_casa", "total_golos_fora",
            "total_golos_casa_depois_previsao", "total_golos_fora_depois_previsao"
        ]
        
        df = convert_numeric_columns(df, numeric_cols)
        
        # 4. Preparar features e targets
        print("\n4️⃣ Preparando features e targets...")
        try:
            X, df, targets = prepare_features_and_targets(df)
            print(f"✅ Features shape: {X.shape}")
            print(f"✅ Targets: {targets}")
        except Exception as e:
            print(f"❌ Erro ao preparar features: {e}")
            traceback.print_exc()
            return
        
        # 5. Carregar modelos e prever
        print("\n5️⃣ Carregando modelos e fazendo previsões...")
        predictions = load_models_and_predict(X, targets, MODELS_DIR)
        
        if not predictions:
            print("❌ Nenhuma previsão foi gerada.")
            return
        
        print(f"✅ Previsões geradas para {len(df)} jogos")
        
        # 6. Criar DataFrame final
        print("\n6️⃣ Criando DataFrame final...")
        df_final = create_final_dataframe(df, predictions)
        
        # 7. Salvar previsões
        print("\n7️⃣ Salvando previsões...")
        save_dataframe(df_final, output_file)
        
        # 8. Processar previsões concatenadas
        print("\n8️⃣ Processando previsões concatenadas...")
        df_loaded = pd.read_csv(output_file, dtype=str).fillna(0)
        
        conceito_cols = [
            "conceito_Mais_0.5_Golos_SegundaParte",
            "conceito_Mais_1.5_Golos_SegundaParte",
            "conceito_Equipa_Perdendo_Marcar_SegundaParte"
        ]
        
        df_loaded = concatenate_sim_predictions(df_loaded, conceito_cols)
        df_loaded = criar_metricas_combinadas_ajustadas(df_loaded)
        
        # 9. Determinar tipo de previsão
        print("\n9️⃣ Determinando tipo de previsão...")
        if os.path.exists(output_concatenado):
            df_existing = pd.read_csv(output_concatenado, dtype=str).fillna(0)
            print(f"   • Arquivo concatenado existente: {len(df_existing)} jogos")
        else:
            df_existing = pd.DataFrame()
            print("   • Nenhum arquivo concatenado existente - criando novo")
        
        # Evitar concatenação com warning
        utc_minus_one = pytz.FixedOffset(-60)
        current_timestamp = datetime.now(utc_minus_one).strftime('%Y-%m-%d %H:%M:%S')
        
        df_loaded['Tipo_Previsao'] = df_loaded['ID_Jogo'].apply(
            lambda game_id: determine_tipo_previsao(game_id, df_existing)
        )
        df_loaded['Timestamp'] = current_timestamp
        
        # 10. Concatenar com existentes
        print("\n🔟 Concatenando com previsões existentes...")
        if not df_existing.empty:
            existing_ids = set(df_existing['ID_Jogo'].astype(str))
            new_ids = set(df_loaded['ID_Jogo'].astype(str))
            df_existing_filtered = df_existing[~df_existing['ID_Jogo'].astype(str).isin(new_ids)]
            
            if not df_existing_filtered.empty:
                print(f"   • Mantendo {len(df_existing_filtered)} jogos antigos não presentes na nova execução")
                # Garantir colunas iguais
                all_cols = set(df_existing_filtered.columns) | set(df_loaded.columns)
                for col in all_cols:
                    if col not in df_existing_filtered.columns:
                        df_existing_filtered[col] = np.nan
                    if col not in df_loaded.columns:
                        df_loaded[col] = np.nan
                
                df_concatenado = pd.concat([df_existing_filtered, df_loaded], ignore_index=True, sort=False)
            else:
                df_concatenado = df_loaded.copy()
                print("   • Todos os jogos antigos estão presentes na nova execução")
        else:
            df_concatenado = df_loaded.copy()
            print("   • Nenhum arquivo anterior para concatenar")
        
        # 11. Salvar resultado final
        save_dataframe(df_concatenado, output_concatenado)
        
        # 12. Deduplicar
        print("\n🧹 Removendo duplicatas...")
        dedupe_keep_latest_only(output_concatenado)
        
        print("\n" + "=" * 70)
        print("✅ PROCESSAMENTO COMPLETO - TODOS OS JOGOS PROCESSADOS!")
        print("=" * 70)
        
        # Mostrar resumo final
        df_final_check = pd.read_csv(output_concatenado, dtype=str).fillna(0)
        
        print(f"\n📊 RESUMO FINAL:")
        print(f"   • Total de jogos processados: {len(df_final_check)}")
        print(f"   • Jogos no input CSV: {len(df)}")
        
        if 'Tipo_Previsao' in df_final_check.columns:
            novos = (df_final_check['Tipo_Previsao'] == 'NOVA PREVISÃO').sum()
            atualizacoes = (df_final_check['Tipo_Previsao'] == 'ATUALIZAÇÃO PREVISÃO').sum()
            print(f"   • NOVAS PREVISÕES: {novos}")
            print(f"   • ATUALIZAÇÕES: {atualizacoes}")
        
        if 'previsao_consensual_ajustada' in df_final_check.columns:
            sim_consensual = (df_final_check['previsao_consensual_ajustada'] == 'Sim').sum()
            perc_consensual = (sim_consensual / len(df_final_check)) * 100 if len(df_final_check) > 0 else 0
            print(f"   • PREVISÕES CONSENSUAIS SIM: {sim_consensual}/{len(df_final_check)} ({perc_consensual:.1f}%)")
        
        if 'nivel_confianca_ajustado' in df_final_check.columns:
            confianca_counts = df_final_check['nivel_confianca_ajustado'].value_counts()
            print(f"   • DISTRIBUIÇÃO DE CONFIANÇA:")
            for nivel, count in confianca_counts.items():
                percent = (count / len(df_final_check)) * 100 if len(df_final_check) > 0 else 0
                print(f"     - {nivel}: {count} ({percent:.1f}%)")
        
        print(f"\n🎯 ARQUIVOS GERADOS:")
        print(f"   • Previsões live: {output_file}")
        print(f"   • Previsões concatenadas: {output_concatenado}")
        
        return {
            'status': 'sucesso',
            'jogos_processados': len(df),
            'jogos_no_concatenado': len(df_final_check),
            'novas_previsoes': novos if 'novos' in locals() else 0,
            'atualizacoes': atualizacoes if 'atualizacoes' in locals() else 0,
            'previsoes_sim': sim_consensual if 'sim_consensual' in locals() else 0
        }

    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        traceback.print_exc()
        return {'status': 'erro', 'mensagem': str(e)}

# Para executar:
resultado = Carregar_modelos_completo()
print(resultado)