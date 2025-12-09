# @title reprocessar_golos_com_registro_efetivo_refactor() - usa CSV mestre e sufixo _depois_previsao (inclui "evolução do Placar_depois_previsao")
def reprocessar_golos_com_registro_efetivo_refactor():
    """
    Refactor:
    - Usa /workspaces/previsao_sofascore/data/df_previsoes_sim_concatenado.csv como CSV mestre
    - Acrescenta o sufixo "_depois_previsao" às colunas geradas
    - Calcula Minutos_jogo_depois_previsao com base na coluna "Inicio"
    - Adiciona coluna "Atualisar" com lógica de controle (0/1)
    """
    try:
        import json, pandas as pd, hashlib, traceback
        from pathlib import Path
        from datetime import datetime, timedelta
        from tqdm import tqdm  # Alterado: tqdm em vez de tqdm.notebook
        import concurrent.futures
        import re

        # ---------------- paths / constantes ----------------
        ROOT = Path('/workspaces/previsao_sofascore')
        BASE = ROOT / 'data' / 'playwright_jsons_full'  # CORRIGIDO: adicionado /data/
        MASTER_CSV_PATH = ROOT / 'data' / 'df_previsoes_sim_concatenado.csv'
        NEW_REGISTRY_PATH = ROOT / 'golos_registro_depois_previsao_hash_filtro.json'
        MAX_WORKERS = 10

        print("=" * 70)
        print("🔄 REPROCESSAMENTO COM REGISTRO EFETIVO (REFACTOR)")
        print("   Inclui 'evolução do Placar_depois_previsao' e 'Minutos_jogo_depois_previsao'")
        print("=" * 70)

        # ---------- Carregar CSV mestre ----------
        if not MASTER_CSV_PATH.exists():
            print(f"❌ {MASTER_CSV_PATH} não encontrado")
            return None

        df_master = pd.read_csv(MASTER_CSV_PATH, dtype=str)
        allowed_ids = set(df_master['ID_Jogo'].dropna().astype(str).tolist())
        print(f"📋 IDs permitidos: {len(allowed_ids)}")
        
        # Verificar se a coluna 'Inicio' existe
        if 'Inicio' not in df_master.columns:
            print("⚠️ Coluna 'Inicio' não encontrada no CSV mestre")
            print(f"   Colunas disponíveis: {list(df_master.columns)}")
            # Procurar coluna alternativa
            if 'Atual_time' in df_master.columns:
                print("   Usando 'Atual_time' como alternativa")
            else:
                print("   ❌ Nenhuma coluna de tempo encontrada")

        # Adicionar coluna "Atualisar" se não existir
        if 'Atualisar' not in df_master.columns:
            df_master['Atualisar'] = '1'  # Inicialmente marcado como já atualizado
            print("✅ Coluna 'Atualisar' adicionada ao dataframe mestre")
        else:
            # Para jogos finalizados que precisam ser atualizados, marcar como 0
            mask_finished_needs_update = (
                (df_master['Tipo_Status'] == 'finished') & 
                (df_master['Atualisar'] == '0')
            )
            if mask_finished_needs_update.any():
                print(f"⚠️  {mask_finished_needs_update.sum()} jogos finalizados marcados para atualização")

        # ---------- Carregar registro de hash ----------
        print("\n📦 Carregando registro de hash (novo)...")
        hash_registry = {}
        if NEW_REGISTRY_PATH.exists():
            try:
                with open(NEW_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                    hash_registry = json.load(f)
                print(f"   • Registro carregado: {len(hash_registry)} jogos registrados")
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar registro: {e}")
                hash_registry = {}
        else:
            print("   • Registro não encontrado - será criado novo")

        # ---------- Encontrar arquivos ----------
        print("\n🔍 Buscando arquivos incidents.json...")
        arquivos = list(BASE.rglob('incidents.json'))
        print(f"   • Total encontrados: {len(arquivos)}")
        
        # Mostrar alguns exemplos
        if arquivos:
            print(f"   • Exemplos: {[str(p.parent.name) for p in arquivos[:3]]}...")
        else:
            print("   ❌ Nenhum arquivo encontrado!")
            print(f"   Caminho procurado: {BASE}")

        # ---------- Decidir o que processar ----------
        gid_para_processar = {}
        gid_skipped = 0
        gid_novos = 0
        gid_atualizados = 0

        def _hash_file(path):
            try:
                with open(path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            except Exception:
                return None

        # Carregar informações do CSV mestre para uso posterior
        master_info = df_master.set_index('ID_Jogo')
        
        for arquivo in arquivos:
            gid = arquivo.parent.name
            if gid not in allowed_ids:
                continue
            
            # Verificar se deve processar baseado em Tipo_Status e Atualisar
            if gid in master_info.index:
                tipo_status = master_info.loc[gid, 'Tipo_Status'] if 'Tipo_Status' in master_info.columns else 'unknown'
                atualisar = master_info.loc[gid, 'Atualisar'] if 'Atualisar' in master_info.columns else '1'
                
                # Lógica de atualização baseada em Tipo_Status
                if tipo_status == 'inprogress':
                    # Jogos em andamento: sempre processar
                    pass  # Continue com o processamento
                elif tipo_status == 'finished':
                    # Jogos finalizados: processar apenas se Atualisar='0'
                    if atualisar == '1':
                        gid_skipped += 1
                        continue
                else:
                    # Outros status: processar
                    pass

            # Calcular hash atual
            try:
                current_hash = _hash_file(arquivo)
                if current_hash is None:
                    continue
            except Exception:
                continue

            # Verificar se precisa processar baseado no hash
            if gid in hash_registry:
                if hash_registry[gid] == current_hash:
                    gid_skipped += 1
                    continue
                else:
                    gid_para_processar[gid] = arquivo
                    gid_atualizados += 1
            else:
                gid_para_processar[gid] = arquivo
                gid_novos += 1

        print(f"\n🎯 ESTATÍSTICAS DE PROCESSAMENTO:")
        print(f"   • Novos jogos: {gid_novos}")
        print(f"   • Atualizações: {gid_atualizados}")
        print(f"   • Pulados (hash igual ou finalizados já atualizados): {gid_skipped}")
        print(f"   • Total a processar: {len(gid_para_processar)}")

        if not gid_para_processar:
            print("\n✅ Nenhum arquivo precisa ser processado!")
            return {
                'novos': 0,
                'atualizados': 0,
                'pulados': gid_skipped,
                'total_arquivos': len(arquivos)
            }

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
            # se já tiver apóstrofo, manter
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
                        minutos = minutos  # Já está correto
                    else:
                        minutos += 45
                    parte = "2ª Parte"
                elif "halftime" in status or "interval" in status or "intervalo" in status:
                    return ("45:00", "Intervalo")
                elif "finished" in status or "finalizado" in status or "ended" in status:
                    return ("90:00", "Finalizado")
                
                return (f"{minutos:02d}:{segundos:02d}", parte)
            except Exception as e:
                print(f"⚠️ Erro ao calcular minutos para {row_info.get('ID_Jogo', 'unknown')}: {e}")
                return ("00:00", "Erro")

        # ---------- Função para processar um jogo ----------
        def processar_jogo(gid_arquivo):
            gid, arquivo = gid_arquivo
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                incidents = data.get('data', {}).get('incidents', [])
                if not incidents:
                    return gid, None, None, None, None

                # ordenar por tempo usando parse_minute_int
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
                evolucao_events = []  # lista de "1-0 (29')"

                for incident in incidents_sorted:
                    incident_type = incident.get('incidentType', '')

                    # Periodos (HT/FT) podem definir scores explícitos
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

                    # Golo -> atualizar running score e registar evento na evolução
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

                        # adicionar evento de evolução no formato "1-0 (29')"
                        evolucao_events.append(f"{home_running}-{away_running} ({minute_txt})")

                # Calcular hash do arquivo (final)
                current_hash = _hash_file(arquivo)

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
                        atualisar_value = '1'  # Jogo finalizado - marcado como atualizado
                    else:
                        atualisar_value = '0'  # Jogo em andamento - precisa atualizar
                else:
                    summary["Minutos_jogo"] = "00:00"
                    summary["Parte"] = "Desconhecido"
                    atualisar_value = '0'
                    tipo_status = 'unknown'

                return gid, summary, current_hash, atualisar_value, tipo_status

            except Exception as e:
                print(f"⚠️ Erro processando jogo {gid}: {e}")
                return gid, None, None, None, None

        # ---------- Processamento paralelo ----------
        print(f"\n⚡ Processando {len(gid_para_processar)} jogos...")
        trabalhos = list(gid_para_processar.items())
        resultados = []
        novos_hashes = {}
        atualisar_values = {}  # Armazenar valores de Atualisar por jogo
        tipo_status_values = {}  # Armazenar Tipo_Status por jogo
        
        stats = {'sucesso': 0, 'erros': 0, 'golos_total': 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(processar_jogo, trabalho): trabalho[0] for trabalho in trabalhos}
            with tqdm(total=len(trabalhos), desc="📊 Processando", dynamic_ncols=True, leave=True) as pbar:
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
                    
                    gid, summary, current_hash, atualisar_value, tipo_status = res

                    if summary:
                        resultados.append(summary)
                        stats['sucesso'] += 1
                        stats['golos_total'] += int(summary.get('quantidade_gols', 0))
                        
                        if current_hash:
                            novos_hashes[gid] = current_hash
                        
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
            # Forçar strings para que nomes com acentos sejam preservados corretamente
            df_novos = pd.DataFrame(resultados, dtype=str)
            
            # Remover possível coluna de índice indesejado
            if '_sort' in df_novos.columns:
                df_novos = df_novos.drop('_sort', axis=1)

            # Criar versão com sufixo "_depois_previsao" para todas as colunas (exceto ID_Jogo)
            cols = [c for c in df_novos.columns if c != 'ID_Jogo']
            rename_map = {c: f"{c}_depois_previsao" for c in cols}
            df_novos = df_novos.rename(columns=rename_map)

            # ---------- Carregar CSV mestre e mesclar ----------
            if MASTER_CSV_PATH.exists():
                try:
                    df_master = pd.read_csv(MASTER_CSV_PATH, dtype=str)
                except Exception as e:
                    print(f"⚠️ Erro ao ler CSV mestre: {e} — criando novo dataframe mestre.")
                    df_master = pd.DataFrame()
            else:
                print("⚠️ CSV mestre não encontrado — será criado novo.")
                df_master = pd.DataFrame()

            # Garantir que ID_Jogo exista no mestre
            if 'ID_Jogo' not in df_master.columns:
                df_master['ID_Jogo'] = []

            # Assegurar que todas as colunas _depois_previsao existam no mestre (criar vazias se necessário)
            for new_col in df_novos.columns:
                if new_col == 'ID_Jogo':
                    continue
                if new_col not in df_master.columns:
                    df_master[new_col] = ""

            # Garantir que coluna Atualisar existe
            if 'Atualisar' not in df_master.columns:
                df_master['Atualisar'] = '1'  # Valor padrão

            # Indexar por ID para atualização eficiente
            df_master = df_master.set_index('ID_Jogo', drop=False)
            df_novos = df_novos.set_index('ID_Jogo', drop=False)

            # Atualizar/Inserir: para cada ID em df_novos, substituir as colunas _depois_previsao no mestre
            for gid in df_novos.index:
                if gid in df_master.index:
                    # atualização: sobrepor as colunas _depois_previsao
                    for col in df_novos.columns:
                        if col == 'ID_Jogo':
                            continue
                        df_master.at[gid, col] = df_novos.at[gid, col]
                    
                    # Atualizar coluna Atualisar com base na lógica
                    if gid in atualisar_values:
                        df_master.at[gid, 'Atualisar'] = atualisar_values[gid]
                else:
                    # inserir nova linha combinando colunas do mestre (vazias) e cols _depois_previsao
                    new_row = {c: "" for c in df_master.columns}
                    new_row['ID_Jogo'] = gid
                    
                    for col in df_novos.columns:
                        if col == 'ID_Jogo':
                            continue
                        new_row[col] = df_novos.at[gid, col]
                    
                    # Definir Atualisar para nova linha
                    if gid in atualisar_values:
                        new_row['Atualisar'] = atualisar_values[gid]
                    else:
                        # Verificar tipo_status para definir Atualisar
                        if gid in tipo_status_values and tipo_status_values[gid] == 'finished':
                            new_row['Atualisar'] = '1'
                        else:
                            new_row['Atualisar'] = '0'
                    
                    df_master = pd.concat([df_master, pd.DataFrame([new_row]).set_index('ID_Jogo')], axis=0)

            # Reset index e garantir ordenação estável
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

            # Separar colunas originais vs novas
            colunas_antes = list(df_master.columns)
            colunas_originais = [c for c in colunas_antes if not c.endswith('_depois_previsao') and c != 'ID_Jogo']
            colunas_novas = [c for c in colunas_antes if c.endswith('_depois_previsao')]

            print(f"📊 Total de colunas no CSV mestre: {len(colunas_antes)}")
            print(f"📋 Colunas originais (preservadas): {len(colunas_originais)}")
            print(f"🆕 Colunas novas com sufixo _depois_previsao: {len(colunas_novas)}")

            # Mostrar exemplos
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

            # Verificar se as colunas específicas foram criadas
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

            # Verificar valores nas colunas importantes
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

        # ---------- Atualizar registro de hash (novo) ----------
        if novos_hashes:
            hash_registry.update(novos_hashes)
            try:
                with open(NEW_REGISTRY_PATH, 'w', encoding='utf-8') as f:
                    json.dump(hash_registry, f, ensure_ascii=False, indent=2)
                print(f"📦 Registro atualizado: {len(novos_hashes)} novos hashes")
            except Exception as e:
                print(f"⚠️ Erro ao salvar registro: {e}")

        # ---------- Relatório final ----------
        print(f"\n" + "=" * 70)
        print("✅ PROCESSAMENTO CONCLUÍDO!")
        print("=" * 70)
        print(f"📊 Estatísticas:")
        print(f"   • Processados com sucesso: {stats['sucesso']}/{len(trabalhos)}")
        print(f"   • Erros: {stats['erros']}")
        print(f"   • Total de golos encontrados: {stats['golos_total']}")
        print(f"   • Novos jogos: {gid_novos}")
        print(f"   • Jogos atualizados: {gid_atualizados}")
        print(f"   • Jogos pulados (sem alterações): {gid_skipped}")
        
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
            'novos': gid_novos,
            'atualizados': gid_atualizados,
            'pulados': gid_skipped,
            'colunas_totais': len(df_master.columns) if 'df_master' in locals() else 0,
            'colunas_novas': len(colunas_novas) if 'colunas_novas' in locals() else 0
        }

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()
        return None

# Para executar:
resultado = reprocessar_golos_com_registro_efetivo_refactor()
print(resultado)