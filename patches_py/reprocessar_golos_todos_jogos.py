# @title reprocessar_golos_todos_jogos() - Processa TODOS os jogos sem filtro de hash
def reprocessar_golos_todos_jogos():
    """
    Versão SEM filtro de hash - Processa TODOS os jogos disponíveis
    - Usa /workspaces/previsao_sofascore/data/df_previsoes_sim_concatenado.csv como CSV mestre
    - Acrescenta o sufixo "_depois_previsao" às colunas geradas
    - Calcula Minutos_jogo_depois_previsao com base na coluna "Inicio"
    - Adiciona coluna "Atualisar" com lógica de controle (0/1)
    - NÃO usa hash para filtrar - processa tudo
    """
    try:
        import json, pandas as pd, traceback
        from pathlib import Path
        from datetime import datetime, timedelta
        from tqdm import tqdm
        import concurrent.futures
        import re

        # ---------------- paths / constantes ----------------
        ROOT = Path('/workspaces/previsao_sofascore')
        BASE = ROOT / 'data' / 'playwright_jsons_full'
        MASTER_CSV_PATH = ROOT / 'data' / 'df_previsoes_sim_concatenado.csv'
        MAX_WORKERS = 10

        print("=" * 70)
        print("🔄 REPROCESSAMENTO COMPLETO - TODOS OS JOGOS (SEM FILTRO DE HASH)")
        print("   Processando TODOS os arquivos incidents.json encontrados")
        print("=" * 70)

        # ---------- Carregar CSV mestre ----------
        if not MASTER_CSV_PATH.exists():
            print(f"❌ {MASTER_CSV_PATH} não encontrado")
            return None

        df_master = pd.read_csv(MASTER_CSV_PATH, dtype=str)
        allowed_ids = set(df_master['ID_Jogo'].dropna().astype(str).tolist())
        print(f"📋 IDs no CSV mestre: {len(allowed_ids)}")
        
        # Verificar se a coluna 'Inicio' existe
        if 'Inicio' not in df_master.columns:
            print("⚠️ Coluna 'Inicio' não encontrada no CSV mestre")
            print(f"   Colunas disponíveis: {list(df_master.columns)}")
            if 'Atual_time' in df_master.columns:
                print("   Usando 'Atual_time' como alternativa")
            else:
                print("   ❌ Nenhuma coluna de tempo encontrada")

        # Adicionar coluna "Atualisar" se não existir
        if 'Atualisar' not in df_master.columns:
            df_master['Atualisar'] = '1'
            print("✅ Coluna 'Atualisar' adicionada ao dataframe mestre")

        # ---------- Encontrar arquivos ----------
        print("\n🔍 Buscando arquivos incidents.json...")
        arquivos = list(BASE.rglob('incidents.json'))
        print(f"   • Total encontrados: {len(arquivos)}")
        
        if arquivos:
            print(f"   • Exemplos: {[str(p.parent.name) for p in arquivos[:3]]}...")
        else:
            print("   ❌ Nenhum arquivo encontrado!")
            print(f"   Caminho procurado: {BASE}")
            return None

        # ---------- Filtrar apenas jogos que estão no CSV mestre ----------
        gid_para_processar = {}
        gid_fora_master = 0
        
        for arquivo in arquivos:
            gid = arquivo.parent.name
            if gid in allowed_ids:
                gid_para_processar[gid] = arquivo
            else:
                gid_fora_master += 1

        print(f"\n🎯 ESTATÍSTICAS DE PROCESSAMENTO:")
        print(f"   • Jogos no CSV mestre: {len(gid_para_processar)}")
        print(f"   • Jogos fora do CSV mestre (ignorados): {gid_fora_master}")
        print(f"   • Total a processar: {len(gid_para_processar)}")

        if not gid_para_processar:
            print("\n❌ Nenhum jogo do CSV mestre encontrado nos arquivos!")
            return None

        # ---------- Auxiliares para minuto/formato ----------
        def parse_minute_int(value):
            """Extrai o minuto inteiro para ordenação (ex: '45+2' -> 45)."""
            try:
                s = str(value)
                m = re.search(r'\d+', s)
                return int(m.group()) if m else 0
            except Exception:
                return 0

        def minute_display(value):
            """Formata o minuto para exibição: '45+2' -> \"45+2'\", 29 -> \"29'\""""
            if value is None:
                return "0'"
            s = str(value).strip()
            if s == '':
                return "0'"
            if s.endswith("'") or s.endswith("’"):
                return s
            else:
                return f"{s}'"

        # Função para calcular minutos do jogo
        def calcular_minutos(row_info):
            try:
                # Tentar obter o início do jogo
                inicio = None
                if 'Inicio' in row_info and pd.notna(row_info['Inicio']):
                    inicio = row_info['Inicio']
                elif 'Atual_time' in row_info and pd.notna(row_info['Atual_time']):
                    inicio = row_info['Atual_time']
                
                if not inicio:
                    return ("00:00", "Desconhecido")
                
                # Obter tipo_status
                tipo_status = row_info.get('Tipo_Status', 'unknown') if 'Tipo_Status' in row_info else 'unknown'
                
                # Se o jogo está finalizado, retornar 90:00
                if tipo_status == 'finished':
                    return ("90:00", "Finalizado")
                
                # Se não tem início válido ou jogo não começou
                try:
                    atual = pd.to_datetime(inicio, utc=True)
                except:
                    return ("00:00", "Não iniciado")
                
                # Calcular diferença para o tempo atual
                agora = datetime.utcnow().replace(tzinfo=None)
                if atual.tzinfo is not None:
                    atual = atual.replace(tzinfo=None)
                
                diff = agora - atual
                if diff.total_seconds() < 0:
                    diff = timedelta(0)
                
                minutos = int(diff.total_seconds() // 60)
                segundos = int(diff.total_seconds() % 60)
                
                if minutos > 130:
                    return ("FT", "Finalizado")
                
                # Verificar período do jogo
                status = str(row_info.get("Status", "")).lower()
                parte = "1ª Parte"
                
                if "2nd" in status or "segundo" in status or "second" in status:
                    if minutos > 45:
                        minutos = minutos
                    else:
                        minutos += 45
                    parte = "2ª Parte"
                elif "halftime" in status or "interval" in status or "intervalo" in status:
                    return ("45:00", "Intervalo")
                elif "finished" in status or "finalizado" in status or "ended" in status:
                    return ("90:00", "Finalizado")
                
                return (f"{minutos:02d}:{segundos:02d}", parte)
            except Exception as e:
                print(f"⚠️ Erro ao calcular minutos: {e}")
                return ("00:00", "Erro")

        # ---------- Função para processar um jogo ----------
        def processar_jogo(gid_arquivo):
            gid, arquivo = gid_arquivo
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                incidents = data.get('data', {}).get('incidents', [])
                if not incidents:
                    return gid, None, None, None

                # ordenar por tempo
                try:
                    incidents_sorted = sorted(incidents, key=lambda x: parse_minute_int(x.get('time', 0)))
                except Exception:
                    incidents_sorted = incidents

                # Processar golos e evolução do placar
                home_goals = []
                away_goals = []
                home_score_ht = 0
                away_score_ht = 0
                home_score_ft = 0
                away_score_ft = 0

                home_running = 0
                away_running = 0
                evolucao_events = []

                for incident in incidents_sorted:
                    incident_type = incident.get('incidentType', '')

                    # Periodos (HT/FT)
                    if incident_type == 'period':
                        time_val = incident.get('time', 0)
                        text_val = str(incident.get('text', '')).upper()
                        home_score = incident.get('homeScore', None)
                        away_score = incident.get('awayScore', None)

                        if (text_val == 'HT') or (parse_minute_int(time_val) == 45):
                            try:
                                if home_score is not None:
                                    home_score_ht = int(home_score)
                                if away_score is not None:
                                    away_score_ht = int(away_score)
                            except Exception:
                                pass

                        if (text_val == 'FT') or (parse_minute_int(time_val) >= 90):
                            try:
                                if home_score is not None:
                                    home_score_ft = int(home_score)
                                if away_score is not None:
                                    away_score_ft = int(away_score)
                            except Exception:
                                pass

                    # Golo
                    if incident_type == 'goal':
                        time_val = incident.get('time', 0)
                        minute_num = parse_minute_int(time_val)
                        minute_txt = minute_display(time_val)

                        is_home = incident.get('isHome', False)

                        if is_home:
                            home_goals.append(minute_num)
                            home_running += 1
                        else:
                            away_goals.append(minute_num)
                            away_running += 1

                        evolucao_events.append(f"{home_running}-{away_running} ({minute_txt})")

                # Criar summary
                summary = {
                    "ID_Jogo": str(gid),
                    "minutos Golos_Casa": ",".join(map(str, home_goals)) if home_goals else "",
                    "minutos Golos_Fora": ",".join(map(str, away_goals)) if away_goals else "",
                    "PLACAR HT": f"{home_score_ht}-{away_score_ht}",
                    "PLACAR FT": f"{home_score_ft}-{away_score_ft}",
                    "total_gols_casa": len(home_goals),
                    "total_golos_fora": len(away_goals),
                    "quantidade_gols": len(home_goals) + len(away_goals),
                    "evolução do Placar": " → ".join(evolucao_events) if evolucao_events else "",
                    "timestamp_processamento": datetime.utcnow().isoformat() + "Z"
                }

                # Obter informações do mestre para calcular minutos
                if gid in master_info.index:
                    row_info = master_info.loc[gid].to_dict()
                    tipo_status = row_info.get('Tipo_Status', 'unknown')
                    
                    # Calcular minutos
                    minutos_str, parte_str = calcular_minutos(row_info)
                    
                    # Adicionar ao summary
                    summary["Minutos_jogo"] = minutos_str
                    summary["Parte"] = parte_str
                    
                    # Determinar valor para Atualisar
                    if tipo_status == 'finished':
                        atualisar_value = '1'
                    else:
                        atualisar_value = '0'
                else:
                    summary["Minutos_jogo"] = "00:00"
                    summary["Parte"] = "Desconhecido"
                    atualisar_value = '0'
                    tipo_status = 'unknown'

                return gid, summary, atualisar_value, tipo_status

            except Exception as e:
                print(f"⚠️ Erro processando jogo {gid}: {e}")
                return gid, None, None, None

        # ---------- Carregar master_info ----------
        master_info = df_master.set_index('ID_Jogo')

        # ---------- Processamento paralelo ----------
        print(f"\n⚡ Processando {len(gid_para_processar)} jogos (SEM FILTRO DE HASH)...")
        trabalhos = list(gid_para_processar.items())
        resultados = []
        atualisar_values = {}
        tipo_status_values = {}
        
        stats = {'sucesso': 0, 'erros': 0, 'golos_total': 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(processar_jogo, trabalho): trabalho[0] for trabalho in trabalhos}
            with tqdm(total=len(trabalhos), desc="📊 Processando TODOS", dynamic_ncols=True, leave=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    res = None
                    try:
                        res = future.result()
                    except Exception as e:
                        print(f"⚠️ Erro no futuro: {e}")
                        res = None
                    
                    pbar.update(1)
                    if not res:
                        stats['erros'] += 1
                        pbar.set_postfix({'Sucesso': stats['sucesso'], 'Golos': stats['golos_total'], 'Erros': stats['erros']})
                        continue
                    
                    gid, summary, atualisar_value, tipo_status = res

                    if summary:
                        resultados.append(summary)
                        stats['sucesso'] += 1
                        stats['golos_total'] += int(summary.get('quantidade_gols', 0))
                        
                        # Armazenar valores para atualização posterior
                        atualisar_values[gid] = atualisar_value
                        tipo_status_values[gid] = tipo_status
                    else:
                        stats['erros'] += 1

                    pbar.set_postfix({
                        'Sucesso': stats['sucesso'],
                        'Golos': stats['golos_total'],
                        'Erros': stats['erros']
                    })

        # ---------- Preparar df_novos com sufixo _depois_previsao ----------
        if resultados:
            df_novos = pd.DataFrame(resultados, dtype=str)
            
            if '_sort' in df_novos.columns:
                df_novos = df_novos.drop('_sort', axis=1)

            # Criar versão com sufixo "_depois_previsao" para todas as colunas (exceto ID_Jogo)
            cols = [c for c in df_novos.columns if c != 'ID_Jogo']
            rename_map = {c: f"{c}_depois_previsao" for c in cols}
            df_novos = df_novos.rename(columns=rename_map)

            # ---------- Carregar CSV mestre e mesclar ----------
            try:
                df_master = pd.read_csv(MASTER_CSV_PATH, dtype=str)
            except Exception as e:
                print(f"⚠️ Erro ao ler CSV mestre: {e} — criando novo dataframe mestre.")
                df_master = pd.DataFrame()

            if 'ID_Jogo' not in df_master.columns:
                df_master['ID_Jogo'] = []

            # Assegurar que todas as colunas _depois_previsao existam no mestre
            for new_col in df_novos.columns:
                if new_col == 'ID_Jogo':
                    continue
                if new_col not in df_master.columns:
                    df_master[new_col] = ""

            # Garantir que coluna Atualisar existe
            if 'Atualisar' not in df_master.columns:
                df_master['Atualisar'] = '1'

            # Indexar por ID para atualização eficiente
            df_master = df_master.set_index('ID_Jogo', drop=False)
            df_novos = df_novos.set_index('ID_Jogo', drop=False)

            # Atualizar/Inserir
            for gid in df_novos.index:
                if gid in df_master.index:
                    # atualização
                    for col in df_novos.columns:
                        if col == 'ID_Jogo':
                            continue
                        df_master.at[gid, col] = df_novos.at[gid, col]
                    
                    # Atualizar coluna Atualisar
                    if gid in atualisar_values:
                        df_master.at[gid, 'Atualisar'] = atualisar_values[gid]
                else:
                    # inserir nova linha
                    new_row = {c: "" for c in df_master.columns}
                    new_row['ID_Jogo'] = gid
                    
                    for col in df_novos.columns:
                        if col == 'ID_Jogo':
                            continue
                        new_row[col] = df_novos.at[gid, col]
                    
                    if gid in atualisar_values:
                        new_row['Atualisar'] = atualisar_values[gid]
                    else:
                        if gid in tipo_status_values and tipo_status_values[gid] == 'finished':
                            new_row['Atualisar'] = '1'
                        else:
                            new_row['Atualisar'] = '0'
                    
                    df_master = pd.concat([df_master, pd.DataFrame([new_row]).set_index('ID_Jogo')], axis=0)

            # Reset index e ordenação
            df_master = df_master.reset_index(drop=True)

            def sort_key(gid):
                try:
                    return int(''.join(filter(str.isdigit, str(gid))))
                except:
                    return float('inf')

            if 'ID_Jogo' in df_master.columns:
                df_master['_sort'] = df_master['ID_Jogo'].apply(sort_key)
                df_master = df_master.sort_values('_sort').reset_index(drop=True)
                df_master = df_master.drop('_sort', axis=1)

            # ==================== VERIFICAÇÃO DE COLUNAS ====================
            print("\n" + "=" * 70)
            print("🔍 VERIFICAÇÃO DE COLUNAS")
            print("=" * 70)

            colunas_antes = list(df_master.columns)
            colunas_originais = [c for c in colunas_antes if not c.endswith('_depois_previsao') and c != 'ID_Jogo']
            colunas_novas = [c for c in colunas_antes if c.endswith('_depois_previsao')]

            print(f"📊 Total de colunas no CSV mestre: {len(colunas_antes)}")
            print(f"📋 Colunas originais (preservadas): {len(colunas_originais)}")
            print(f"🆕 Colunas novas com sufixo _depois_previsao: {len(colunas_novas)}")

            print("\n📋 Exemplos de colunas originais (primeiras 10):")
            for col in colunas_originais[:10]:
                print(f"   • {col}")
            if len(colunas_originais) > 10:
                print(f"   ... e mais {len(colunas_originais) - 10} colunas")

            print("\n🆕 Colunas novas criadas/atualizadas:")
            for col in sorted(colunas_novas):
                print(f"   • {col}")

            print(f"\n📈 IDs processados nesta execução: {len(df_novos)}")
            print(f"📈 Total de registros no CSV mestre: {len(df_master)}")

            # Verificar colunas específicas
            colunas_verificar = [
                'minutos Golos_Casa_depois_previsao',
                'minutos Golos_Fora_depois_previsao',
                'PLACAR HT_depois_previsao',
                'PLACAR FT_depois_previsao',
                'evolução do Placar_depois_previsao',
                'Minutos_jogo_depois_previsao',
                'Parte_depois_previsao',
                'Atualisar'
            ]

            print("\n✅ STATUS DAS COLUNAS-CHAVE:")
            for col in colunas_verificar:
                if col in df_master.columns:
                    print(f"   ✓ {col} - PRESENTE")
                else:
                    print(f"   ✗ {col} - AUSENTE")

            # Verificar valores
            print("\n📝 EXEMPLOS DE VALORES:")
            
            if 'evolução do Placar_depois_previsao' in df_master.columns:
                exemplos = df_master[df_master['evolução do Placar_depois_previsao'].notna() &
                                    (df_master['evolução do Placar_depois_previsao'] != '')]
                if len(exemplos) > 0:
                    print("   'evolução do Placar_depois_previsao':")
                    for i, (idx, row) in enumerate(exemplos.head(2).iterrows()):
                        print(f"     ID {row['ID_Jogo']}: {row['evolução do Placar_depois_previsao'][:80]}...")
            
            if 'Minutos_jogo_depois_previsao' in df_master.columns:
                exemplos = df_master[df_master['Minutos_jogo_depois_previsao'].notna() &
                                    (df_master['Minutos_jogo_depois_previsao'] != '')]
                if len(exemplos) > 0:
                    print("   'Minutos_jogo_depois_previsao':")
                    for i, (idx, row) in enumerate(exemplos.head(2).iterrows()):
                        print(f"     ID {row['ID_Jogo']}: {row['Minutos_jogo_depois_previsao']} ({row.get('Parte_depois_previsao', '')})")
            
            if 'Atualisar' in df_master.columns:
                counts = df_master['Atualisar'].value_counts()
                print(f"   'Atualisar' - Distribuição:")
                for valor, quantidade in counts.items():
                    print(f"     '{valor}': {quantidade} jogos")

            # Salvar mestre atualizado
            df_master.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"\n💾 CSV mestre salvo/atualizado: {MASTER_CSV_PATH}")

        else:
            print("\nℹ️ Nenhum resultado processado para gravar no CSV mestre.")

        # ---------- Relatório final ----------
        print(f"\n" + "=" * 70)
        print("✅ PROCESSAMENTO COMPLETO - TODOS OS JOGOS!")
        print("=" * 70)
        print(f"📊 Estatísticas:")
        print(f"   • Processados com sucesso: {stats['sucesso']}/{len(trabalhos)}")
        print(f"   • Erros: {stats['erros']}")
        print(f"   • Total de golos encontrados: {stats['golos_total']}")
        print(f"   • Jogos no CSV mestre: {len(gid_para_processar)}")
        print(f"   • Jogos fora do CSV mestre (ignorados): {gid_fora_master}")
        
        # Estatísticas de Atualisar
        if 'df_master' in locals():
            if 'Atualisar' in df_master.columns:
                atualisar_0 = (df_master['Atualisar'] == '0').sum()
                atualisar_1 = (df_master['Atualisar'] == '1').sum()
                print(f"\n📋 Estatísticas de Atualisar:")
                print(f"   • Atualisar='0' (precisa atualizar): {atualisar_0}")
                print(f"   • Atualisar='1' (já atualizado): {atualisar_1}")

        return {
            'processados': stats['sucesso'],
            'erros': stats['erros'],
            'total_golos': stats['golos_total'],
            'jogos_no_master': len(gid_para_processar),
            'jogos_fora_master': gid_fora_master,
            'colunas_totais': len(df_master.columns) if 'df_master' in locals() else 0,
            'colunas_novas': len(colunas_novas) if 'colunas_novas' in locals() else 0
        }

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()
        return None

# Para executar:
resultado = reprocessar_golos_todos_jogos()
print(resultado)