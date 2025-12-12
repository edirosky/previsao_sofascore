# @title tarefa7_refactor_usando_depois_previsao() — timing dos golos + análise (prefere *_depois_previsao)
import pandas as pd
import os
import numpy as np
import re
from IPython.display import display
from pathlib import Path

def Analise_metricas_confianca():
    try:
        minutos_base = 46  # @param {"type":"integer"}
        
        # Caminho fixo para o CSV mestre
        MASTER_CSV_PATH = Path('/workspaces/previsao_sofascore/scripts/data/df_previsoes_sim_concatenado.csv')
        
        other_thresholds = [50, 55, 60, 65, 70, 75, 80, 85]
        thresholds = [minutos_base] + [t for t in other_thresholds if t != minutos_base]

        if not MASTER_CSV_PATH.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {MASTER_CSV_PATH}")

        df = pd.read_csv(MASTER_CSV_PATH, dtype=str)
        print(f"📊 CSV carregado com {len(df)} linhas.\n")

        # --- helpers para resolver colunas com variantes e sufixos (_depois_previsao / _atual / _ajustado / underscores) ---
        def variants(name):
            # gera variantes comuns para procurar no dataframe
            # Prioridade: _depois_previsao, depois _atual, depois _ajustado, depois sem sufixo
            n_space = name
            n_underscore = name.replace(' ', '_')
            
            # Lista de sufixos em ordem de prioridade
            sufixos = ['_depois_previsao', '_atual', '_ajustado', '']
            
            v = []
            for sufixo in sufixos:
                v.append(f"{n_space}{sufixo}")
                v.append(f"{n_underscore}{sufixo}")
            
            # Adicionar também variantes sem espaços para colunas como "evolução do Placar"
            n_no_space = name.replace(' ', '')
            for sufixo in sufixos:
                v.append(f"{n_no_space}{sufixo}")
            
            # garantir unicidade mantendo ordem
            seen = set()
            out = []
            for x in v:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        def col_get_series(series, name, default=''):
            # retorna o primeiro valor não-nulo entre as variantes solicitadas para a linha (Series)
            for c in variants(name):
                # use .get para evitar KeyError; Series.get retorna None se não existir
                val = series.get(c, None)
                if pd.notna(val) and val != '':
                    return val
            return default

        # versões específicas usadas no código (mantêm a intenção original)
        def get_status_field(row):
            return str(col_get_series(row, 'Status', '')).strip()

        def get_minutos_jogo_field(row):
            return str(col_get_series(row, 'Minutos_jogo', '')).strip()

        def get_evolucao_field(row):
            # 'evolução do Placar' é nome com espaço/acentuação no teu df original
            return str(col_get_series(row, 'evolução do Placar', '')).strip()

        def get_previsao_sim_concat(row):
            return str(col_get_series(row, 'Previsao_Sim_Concatenado', '')).strip()

        def get_nivel_confianca(row):
            return str(col_get_series(row, 'nivel_confianca', '')).strip()

        def get_previsao_consensual(row):
            return str(col_get_series(row, 'previsao_consensual', '')).strip()

        def get_concordancia(row):
            return str(col_get_series(row, 'concordancia', '')).strip()

        # parser de minutos (idêntico ao teu original, robusto a "45+2" etc.)
        def parse_goal_minutes(evolucao_str):
            if not evolucao_str or not isinstance(evolucao_str, str) or evolucao_str.strip() == '' or evolucao_str.strip().upper() == 'N/D':
                return []
            minutos = []
            eventos = [e.strip() for e in str(evolucao_str).split('→') if e.strip()]
            for evento in eventos:
                if '(' in evento and ')' in evento:
                    inside = evento.split('(')[1].split(')')[0]
                else:
                    inside = evento
                inside = inside.replace("'", "").replace('"', '').strip()
                m = re.search(r'(\d+(?:\+\d+)?)', inside)
                if m:
                    token = m.group(1)
                    if '+' in token:
                        parts = token.split('+')
                        try:
                            val = float(parts[0]) + float(parts[1])
                        except:
                            try:
                                val = float(parts[0])
                            except:
                                continue
                    else:
                        try:
                            val = float(token)
                        except:
                            continue
                    minutos.append(val)
                else:
                    digits = re.findall(r'\d+', inside)
                    if digits:
                        try:
                            minutos.append(float(digits[0]))
                        except:
                            pass
            return minutos

        # detectar jogo terminado
        def is_ended_row(row):
            status_raw = get_status_field(row)
            status_norm = str(status_raw).strip().lower()
            ended_set = {'finished', 'ended', 'ft', 'final', 'aet', 'ap', 'terminado', 'full-time', 'full time'}
            if status_norm in ended_set:
                return True
            
            min_jogo_raw = get_minutos_jogo_field(row)
            if isinstance(min_jogo_raw, str) and min_jogo_raw.strip().lower() in {'terminado', 'term', 'ended', 'finished'}:
                return True
            
            try:
                if isinstance(min_jogo_raw, str) and min_jogo_raw.strip() != '':
                    m = re.search(r'(\d+)', min_jogo_raw)
                    if m:
                        val = float(m.group(1))
                        if val >= 90:
                            return True
            except Exception:
                pass
            
            # Verificar também por tipo_status se existir
            tipo_status = str(col_get_series(row, 'Tipo_Status', '')).strip().lower()
            if tipo_status == 'finished':
                return True
                
            return False

        # status por threshold
        def status_for_threshold(row, threshold):
            terminado = is_ended_row(row)
            evolucao = get_evolucao_field(row)
            minutos = parse_goal_minutes(evolucao)
            minutos_apos = [m for m in minutos if (m > 45 and m > threshold)]
            if terminado:
                return 'green' if len(minutos_apos) >= 1 else 'red'
            else:
                return 'green' if len(minutos_apos) >= 1 else 'pendente'

        # Criar coluna geral se não existir
        if 'Estado_Previsao_Geral' not in df.columns:
            df['Estado_Previsao_Geral'] = ''

        # Preparar colunas por threshold (cria apenas se não existirem)
        for t in thresholds:
            for colname, default in [
                (f"Estado_{t}", ""),
                (f"Golos_apos_{t}", ""),
                (f"Bucket_{t}", ""),
                (f"Min_pri_golo_apos_{t}", np.nan)
            ]:
                if colname not in df.columns:
                    df[colname] = default

        if 'Estado_Previsao_Detalhado' not in df.columns:
            df['Estado_Previsao_Detalhado'] = ""

        # Stats expandidos
        stats = {t: {'green':0, 'red':0, 'pendente':0, 'analisadas':0,
                     'resolved_goals_counts':[], 'resolved_first_minutes':[],
                     'resolved_total_goals_after_threshold':0,
                     'resolved_total_goals_second_half':0,
                     'resolved_goals_between_46_70':0, 'resolved_goals_between_46_80':0,
                     'resolved_games_with_goal_between_46_70':0, 'resolved_games_with_goal_between_46_80':0,
                     'green_high_confidence':0, 'green_medium_confidence':0, 'green_low_confidence':0,
                     'red_high_confidence':0, 'red_medium_confidence':0, 'red_low_confidence':0,
                     'accuracy_by_confidence': {}}
                 for t in thresholds}

        total_linhas_analisadas = 0

        # Loop principal
        for idx, row in df.iterrows():
            previsoes_str = get_previsao_sim_concat(row)
            has_target = False
            detalhes = []
            geral_estado = 'pendente'

            if isinstance(previsoes_str, str) and previsoes_str.strip():
                lista_previsoes = [p.strip() for p in previsoes_str.split(';') if p.strip()]
                for previsao in lista_previsoes:
                    if previsao.startswith('Mais_0.5_Golos_SegundaParte'):
                        has_target = True
                        estado_base = status_for_threshold(row, minutos_base)
                        detalhes.append(f"{previsao}: {estado_base}")
                        if estado_base == 'green':
                            geral_estado = 'green'
                        elif estado_base == 'red' and geral_estado == 'pendente':
                            geral_estado = 'red'

                if has_target:
                    total_linhas_analisadas += 1
                    evolucao = get_evolucao_field(row)
                    minutos_all = parse_goal_minutes(evolucao)
                    minutos_second_half = [m for m in minutos_all if m > 45]

                    for t in thresholds:
                        estado_t = status_for_threshold(row, t)
                        stats[t]['analisadas'] += 1
                        stats[t][estado_t] = stats[t].get(estado_t, 0) + 1
                        df.at[idx, f"Estado_{t}"] = estado_t

                        minutos_apos = [m for m in minutos_second_half if m > t]
                        golos_apos = len(minutos_apos)
                        df.at[idx, f"Golos_apos_{t}"] = golos_apos

                        if golos_apos == 0:
                            bucket = '0 golos'
                        elif golos_apos == 1:
                            bucket = '1 gol'
                        elif golos_apos == 2:
                            bucket = '2 golos'
                        else:
                            bucket = '3+ golos'
                        df.at[idx, f"Bucket_{t}"] = bucket

                        if minutos_apos:
                            df.at[idx, f"Min_pri_golo_apos_{t}"] = float(min(minutos_apos))
                        else:
                            df.at[idx, f"Min_pri_golo_apos_{t}"] = np.nan

                        is_resolved = is_ended_row(row)
                        if is_resolved:
                            stats[t]['resolved_goals_counts'].append(golos_apos)
                            if minutos_apos:
                                stats[t]['resolved_first_minutes'].append(min(minutos_apos))
                            stats[t]['resolved_total_goals_after_threshold'] += golos_apos

                            num_second_half_goals = len(minutos_second_half)
                            stats[t]['resolved_total_goals_second_half'] += num_second_half_goals

                            before70 = sum(1 for m in minutos_second_half if (m <= 70))
                            before80 = sum(1 for m in minutos_second_half if (m <= 80))
                            stats[t]['resolved_goals_between_46_70'] += before70
                            stats[t]['resolved_goals_between_46_80'] += before80

                            if before70 > 0:
                                stats[t]['resolved_games_with_goal_between_46_70'] += 1
                            if before80 > 0:
                                stats[t]['resolved_games_with_goal_between_46_80'] += 1

                            # novas métricas de confiança (procura variantes _depois_previsao primeiro)
                            nivel_confianca = col_get_series(row, 'nivel_confianca', '')
                            previsao_consensual = col_get_series(row, 'previsao_consensual', '')

                            if estado_t == 'green':
                                if 'Alta' in nivel_confianca:
                                    stats[t]['green_high_confidence'] += 1
                                elif 'Média' in nivel_confianca:
                                    stats[t]['green_medium_confidence'] += 1
                                elif 'Baixa' in nivel_confianca or 'Muito Baixa' in nivel_confianca:
                                    stats[t]['green_low_confidence'] += 1
                            elif estado_t == 'red':
                                if 'Alta' in nivel_confianca:
                                    stats[t]['red_high_confidence'] += 1
                                elif 'Média' in nivel_confianca:
                                    stats[t]['red_medium_confidence'] += 1
                                elif 'Baixa' in nivel_confianca or 'Muito Baixa' in nivel_confianca:
                                    stats[t]['red_low_confidence'] += 1

                    df.at[idx, 'Estado_Previsao_Detalhado'] = ' | '.join(detalhes) if detalhes else ''
                    df.at[idx, 'Estado_Previsao_Geral'] = status_for_threshold(row, minutos_base)
                else:
                    # sem target: garantir colunas vazias/set default
                    for t in thresholds:
                        df.at[idx, f"Estado_{t}"] = ''
                        df.at[idx, f"Golos_apos_{t}"] = ''
                        df.at[idx, f"Bucket_{t}"] = ''
                        df.at[idx, f"Min_pri_golo_apos_{t}"] = np.nan
                    df.at[idx, 'Estado_Previsao_Detalhado'] = ''
                    if not df.at[idx, 'Estado_Previsao_Geral']:
                        df.at[idx, 'Estado_Previsao_Geral'] = 'pendente'
            else:
                # sem previsoes: garantir colunas vazias/set default
                for t in thresholds:
                    df.at[idx, f"Estado_{t}"] = ''
                    df.at[idx, f"Golos_apos_{t}"] = ''
                    df.at[idx, f"Bucket_{t}"] = ''
                    df.at[idx, f"Min_pri_golo_apos_{t}"] = np.nan
                df.at[idx, 'Estado_Previsao_Detalhado'] = ''
                if not df.at[idx, 'Estado_Previsao_Geral']:
                    df.at[idx, 'Estado_Previsao_Geral'] = 'pendente'

        # Salvar CSV (sobrepõe o mestre)
        df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ CSV atualizado com análise de previsões salvo em: {MASTER_CSV_PATH}\n")

        # ========= ANÁLISES E RELATÓRIOS =========
        print("🎯 ANÁLISE DAS MÉTRICAS COMBINADAS E CONFIANÇA")
        print("=" * 60)

        # Preparar df_resolvidos se possível
        df_resolvidos = df[df['Estado_Previsao_Geral'].isin(['green', 'red'])] if 'Estado_Previsao_Geral' in df.columns else pd.DataFrame()

        # Verificar se temos colunas de confiança
        confianca_cols = [c for c in df.columns if 'confianca' in c.lower()]
        if confianca_cols:
            print(f"📋 Colunas de confiança encontradas: {confianca_cols}")

        # Performance da Previsão Consensual (buscar variantes)
        consensual_cols = [c for c in df.columns if 'consensual' in c.lower()]
        if consensual_cols and len(df_resolvidos) > 0:
            # Usar a primeira coluna consensual encontrada
            consensual_col = consensual_cols[0]
            mask_consensual_sim = df_resolvidos[consensual_col] == 'Sim'
            mask_green = df_resolvidos['Estado_Previsao_Geral'] == 'green'

            acertos_consensual = len(df_resolvidos[mask_consensual_sim & mask_green])
            total_consensual_sim = len(df_resolvidos[mask_consensual_sim])
            accuracy_consensual = (acertos_consensual / total_consensual_sim * 100) if total_consensual_sim > 0 else 0

            print(f"📊 Performance da Previsão Consensual ({consensual_col}):")
            print(f"   Acertos: {acertos_consensual}/{total_consensual_sim} ({accuracy_consensual:.1f}%)")

        # Performance por Nível de Confiança
        if confianca_cols and len(df_resolvidos) > 0:
            print(f"\n📈 Performance por Nível de Confiança:")
            confianca_col = confianca_cols[0]
            for nivel in ['Alta Confiança', 'Média Confiança', 'Baixa Confiança', 'Muito Baixa Confiança']:
                mask_nivel = df_resolvidos[confianca_col] == nivel
                mask_green_nivel = mask_nivel & (df_resolvidos['Estado_Previsao_Geral'] == 'green')
                total_nivel = len(df_resolvidos[mask_nivel])

                if total_nivel > 0:
                    acertos_nivel = len(df_resolvidos[mask_green_nivel])
                    accuracy_nivel = (acertos_nivel / total_nivel * 100)
                    print(f"   {nivel}: {acertos_nivel}/{total_nivel} ({accuracy_nivel:.1f}%)")

        # Concordância (se existir)
        concordancia_cols = [c for c in df.columns if 'concordancia' in c.lower()]
        if concordancia_cols and len(df_resolvidos) > 0:
            print(f"\n🤝 Análise de Concordância entre Métricas:")
            concordancia_col = concordancia_cols[0]
            for concordancia in range(6):  # 0 a 5
                mask_concord = df_resolvidos[concordancia_col] == str(concordancia)
                total_concord = len(df_resolvidos[mask_concord])

                if total_concord > 0:
                    acertos_concord = len(df_resolvidos[mask_concord & (df_resolvidos['Estado_Previsao_Geral'] == 'green')])
                    accuracy_concord = (acertos_concord / total_concord * 100)
                    print(f"   {concordancia}/5 métricas: {acertos_concord}/{total_concord} ({accuracy_concord:.1f}%)")

        # Timing summary (igual ao original)
        print("\n" + "="*60)
        print("⏰ ESTUDOS DE TIMING DOS GOLOS (2ª parte: 46–70 e 46–80)")
        print("="*60)
        print(f"Total de linhas com previsões 'Mais_0.5_Golos_SegundaParte' analisadas: {total_linhas_analisadas}\n")

        def fmt_pct(part, total):
            if total == 0:
                return "0.0%"
            return f"{(part/total*100):.1f}%"

        for t in thresholds:
            s = stats[t]
            resolved_games = s.get('green', 0) + s.get('red', 0)
            pct_green = fmt_pct(s.get('green', 0), resolved_games) if resolved_games > 0 else "0.0%"
            pct_red = fmt_pct(s.get('red', 0), resolved_games) if resolved_games > 0 else "0.0%"

            bucket_counts = {'1 gol':0, '2 golos':0, '3+ golos':0, '0 golos':0}
            for cnt in s['resolved_goals_counts']:
                if cnt == 0:
                    bucket_counts['0 golos'] += 1
                elif cnt == 1:
                    bucket_counts['1 gol'] += 1
                elif cnt == 2:
                    bucket_counts['2 golos'] += 1
                else:
                    bucket_counts['3+ golos'] += 1
            total_resolved_games = len(s['resolved_goals_counts'])

            total_goals_after_threshold = s['resolved_total_goals_after_threshold']
            total_goals_second_half = s['resolved_total_goals_second_half']
            total_before70_goals = s['resolved_goals_between_46_70']
            total_before80_goals = s['resolved_goals_between_46_80']

            pct_goals_before70 = fmt_pct(total_before70_goals, total_goals_second_half) if total_goals_second_half > 0 else "0.0%"
            pct_goals_before80 = fmt_pct(total_before80_goals, total_goals_second_half) if total_goals_second_half > 0 else "0.0%"

            pct_games_with_before70 = fmt_pct(s['resolved_games_with_goal_between_46_70'], total_resolved_games) if total_resolved_games > 0 else "0.0%"
            pct_games_with_before80 = fmt_pct(s['resolved_games_with_goal_between_46_80'], total_resolved_games) if total_resolved_games > 0 else "0.0%"

            first_minutes = np.array(s['resolved_first_minutes']) if s['resolved_first_minutes'] else np.array([])
            mean_first = f"{np.mean(first_minutes):.1f}" if first_minutes.size > 0 else "N/D"
            median_first = f"{np.median(first_minutes):.1f}" if first_minutes.size > 0 else "N/D"
            std_first = f"{np.std(first_minutes, ddof=0):.1f}" if first_minutes.size > 0 else "N/D"

            print(f"--- Threshold > {t} minutos ---")
            print(f"Analisadas (linhas com previsao): {s['analisadas']}")
            print(f"Green: {s.get('green',0)} | Red: {s.get('red',0)} | Pendente: {s.get('pendente',0)}")
            print(f"Resolvidos (Green+Red): {resolved_games}")
            print(f"% Green (sobre resolvidos): {pct_green}")
            print(f"% Red   (sobre resolvidos): {pct_red}")

            if resolved_games > 0:
                total_green = s.get('green', 0)
                green_high = s.get('green_high_confidence', 0)
                green_medium = s.get('green_medium_confidence', 0)
                green_low = s.get('green_low_confidence', 0)

                print(f"Green por Confiança - Alta: {green_high}, Média: {green_medium}, Baixa: {green_low}")

                if total_green > 0:
                    print(f"Distribuição Green - Alta: {fmt_pct(green_high, total_green)}, "
                          f"Média: {fmt_pct(green_medium, total_green)}, "
                          f"Baixa: {fmt_pct(green_low, total_green)}")

            print(f"Distribuição por quantidade (resolvidos = {total_resolved_games} jogos):")
            print(f"0 golos: {bucket_counts['0 golos']} | 1 gol: {bucket_counts['1 gol']} | "
                  f"2 golos: {bucket_counts['2 golos']} | 3+ golos: {bucket_counts['3+ golos']}")

            print(f"Timing dos golos (2ª parte):")
            print(f"Total golos 2ª parte: {total_goals_second_half}")
            print(f"Golos 46-70: {total_before70_goals} ({pct_goals_before70}) | "
                  f"Golos 46-80: {total_before80_goals} ({pct_goals_before80})")
            print(f"Jogos com golos 46-70: {pct_games_with_before70} | "
                  f"Jogos com golos 46-80: {pct_games_with_before80}")

            print(f"Primeiro golo após {t}min: Média: {mean_first}, Mediana: {median_first}\n")

        # Estatísticas gerais
        total_green = len(df[df['Estado_Previsao_Geral'] == 'green'])
        total_red = len(df[df['Estado_Previsao_Geral'] == 'red'])
        total_pendente = len(df[df['Estado_Previsao_Geral'] == 'pendente'])
        total_resolved = total_green + total_red
        pct_green_overall = fmt_pct(total_green, total_resolved) if total_resolved>0 else "0.0%"
        pct_red_overall = fmt_pct(total_red, total_resolved) if total_resolved>0 else "0.0%"

        print("=== ESTATÍSTICAS GERAIS DAS PREVISÕES ===")
        print(f"Total previsões analisadas: {total_linhas_analisadas}")
        print(f"Green: {total_green} | Red: {total_red} | Pendente: {total_pendente}")
        print(f"Resolvidos: {total_resolved}")
        print(f"% Green: {pct_green_overall} | % Red: {pct_red_overall}")

        # Amostra rápida
        sample_cols = [c for c in df.columns if any(prefix in c for prefix in
                      ['Estado_', 'Golos_apos_', 'Bucket_', 'Min_pri_golo_apos_',
                       'previsao_consensual', 'nivel_confianca', 'concordancia'])]
        sample_cols = sample_cols[:15]  # Limitar para visualização

        print("\n=== AMOSTRA DAS COLUNAS ANALÍTICAS ===")
        if sample_cols:
            print(df[sample_cols].head(8).to_string(index=False))
            if 'IPython.display' in globals():
                display(df[sample_cols].head(8))
        else:
            print("Nenhuma coluna analítica encontrada para exibir.")
        print(f"\n✅ Análise concluída! {len(df)} jogos processados.")

    except Exception as e:
        import traceback
        print("❌ Erro no processamento da tarefa7:")
        traceback.print_exc()

# executar
Analise_metricas_confianca()