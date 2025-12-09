# ⚽ Frontend Streamlit para Previsões SofaScore
# Versão refatorada com abas separadas e melhorias de UX
# MODIFICAÇÃO: Jogos com Tipo_Status_depois_previsao != "inprogress" vão para aba Terminados

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Configuração da página - DEVE SER A PRIMEIRA COISA
st.set_page_config(
    page_title="Previsões SofaScore",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar caminho para importações
sys.path.append(str(Path(__file__).parent.parent))

# CSS customizado melhorado
st.markdown("""
<style>
    /* Estilos principais */
    .stApp {
        max-width: 100%;
        padding: 0.5rem 1rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.5rem;
    }
    
    .game-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .game-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
        transform: translateY(-1px);
    }
    
    /* Seções de dados */
    .data-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        border-left: 4px solid #94a3b8;
    }
    
    .data-section-title {
        color: #475569;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    /* Cores de confiança */
    .confidence-high { 
        background: #10B98120;
        color: #10B981 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-medium { 
        background: #F59E0B20;
        color: #F59E0B !important; 
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-low { 
        background: #EF444420;
        color: #EF4444 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-very-low { 
        background: #6B728020;
        color: #6B7280 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* Cores de estado */
    .estado-green { 
        background: #10B98120;
        color: #10B981 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .estado-red { 
        background: #EF444420;
        color: #EF4444 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .estado-pendente { 
        background: #6B728020;
        color: #6B7280 !important; 
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* Status colors */
    .status-live { 
        background: #10B98120;
        color: #10B981;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .status-finished { 
        background: #6B728020;
        color: #6B7280;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .status-cancelled { 
        background: #EF444420;
        color: #EF4444;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* Botões e inputs */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    /* Tabelas compactas */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.85rem;
    }
    
    /* Tabs compactas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F8FAFC;
        border-radius: 4px 4px 0 0;
        padding: 8px 12px;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Espaçamentos reduzidos */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Badges compactos */
    .badge {
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 12px;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    
    /* Grid de métricas */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class DataManager:
    """Gerencia o carregamento e processamento dos dados"""
    
    def __init__(self):
        # Caminhos relativos para funcionar no Codespaces
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_path = self.base_dir / "data" / "df_previsoes_sim_concatenado.csv"
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Carrega os dados do arquivo CSV"""
        try:
            if self.data_path.exists():
                # Ler o arquivo CSV
                self.data = pd.read_csv(self.data_path, dtype=str, low_memory=False)
                
                # Limpar dados
                self.data = self.clean_data(self.data)
                
                # Processar dados
                self.data = self.process_data(self.data)
                
                return True
            else:
                st.error(f"❌ Arquivo não encontrado: {self.data_path}")
                return False
                    
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def clean_data(self, df):
        """Limpa os dados básicos"""
        if df.empty:
            return df
        
        # Substituir valores nulos
        df = df.replace(['', 'nan', 'NaN', 'None', 'null', 'NA'], np.nan)
        
        # Converter colunas numéricas importantes
        numeric_cols = ['Placar_Home', 'Placar_Away', 'total_golos_casa', 'total_golos_fora']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def process_data(self, df):
        """Processa os dados para visualização"""
        if df.empty:
            return df
        
        # Converter Timestamp
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df['Data_Hora'] = df['Timestamp'].dt.strftime('%d/%m/%Y %H:%M')
                df['Hora'] = df['Timestamp'].dt.strftime('%H:%M')
                df['Data'] = df['Timestamp'].dt.strftime('%d/%m/%Y')
            except:
                pass
        
        # Criar coluna de status simplificado
        if 'Status' in df.columns:
            df['Status_Simples'] = df['Status'].apply(self.simplify_status)
        
        # Extrair probabilidades numéricas
        proba_cols = [col for col in df.columns if '_proba' in col and col.startswith('pred_')]
        for col in proba_cols:
            try:
                numeric_col = f"{col}_num"
                df[numeric_col] = df[col].astype(str).str.replace('%', '').astype(float).fillna(0)
            except:
                pass
        
        # Ordenar por Timestamp (mais recente primeiro)
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp', ascending=False)
        
        return df
    
    @staticmethod
    def simplify_status(status):
        """Simplifica o status do jogo"""
        if pd.isna(status):
            return "Desconhecido"
        
        status_str = str(status).lower()
        
        if '1st' in status_str or '1ª' in status_str:
            return "1ª Parte"
        elif '2nd' in status_str or '2ª' in status_str:
            return "2ª Parte"
        elif 'finished' in status_str or 'finalizado' in status_str:
            return "Finalizado"
        elif 'inprogress' in status_str or 'andamento' in status_str:
            return "Em Andamento"
        elif 'canceled' in status_str or 'cancelled' in status_str:
            return "Cancelado"
        elif 'notstarted' in status_str:
            return "Não Iniciado"
        elif 'postponed' in status_str:
            return "Adiado"
        else:
            return status_str.capitalize()

def display_header():
    """Exibe o cabeçalho da aplicação"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown('<h1 style="margin-bottom: 0.5rem;">⚽ Previsões SofaScore</h1>', unsafe_allow_html=True)
        st.markdown('<p style="margin: 0;">Sistema Inteligente de Previsão de Futebol - Análise em Tempo Real</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.write("")
        st.write("")
        atualizar = st.button("🔄 Atualizar", use_container_width=True)
        if atualizar:
            st.rerun()
    
    with col3:
        st.write("")
        st.write("")
        agora = datetime.now().strftime('%H:%M')
        st.metric("Hora Atual", agora)

def display_sidebar(data_manager):
    """Cria a sidebar com filtros e informações"""
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        # Informações do sistema
        st.markdown("### 📊 Estatísticas")
        if data_manager.data is not None and not data_manager.data.empty:
            total_jogos = len(data_manager.data)
            
            # Contar jogos live baseado na nova coluna
            if 'Tipo_Status_depois_previsao' in data_manager.data.columns:
                live_jogos = len(data_manager.data[
                    data_manager.data['Tipo_Status_depois_previsao'].fillna('').str.lower() == 'inprogress'
                ])
                finalizados = total_jogos - live_jogos
            else:
                # Fallback para lógica antiga
                live_jogos = len(data_manager.data[data_manager.data['Status_Simples'].isin(['1ª Parte', '2ª Parte', 'Em Andamento'])])
                finalizados = len(data_manager.data[data_manager.data['Status_Simples'] == 'Finalizado'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total_jogos)
            with col2:
                st.metric("Live", live_jogos)
        
        # Filtros
        st.markdown("### 🔍 Filtros")
        
        if data_manager.data is None or data_manager.data.empty:
            st.info("Sem dados para filtrar")
            return {}
        
        filters = {}
        
        # Filtro por torneio
        if 'Torneio' in data_manager.data.columns:
            torneios = ['Todos'] + sorted(data_manager.data['Torneio'].dropna().unique().tolist())
            filters['torneio'] = st.selectbox("🎯 Torneio", torneios)
        
        # Filtro por estado da previsão
        if 'Estado_Previsao_Geral' in data_manager.data.columns:
            estado_opts = ['Todos', 'green', 'red', 'pendente']
            filters['estado'] = st.selectbox("📈 Estado", estado_opts)
        
        # Filtro por confiança
        if 'nivel_confianca_ajustado' in data_manager.data.columns:
            confianca_opts = ['Todos'] + sorted(data_manager.data['nivel_confianca_ajustado'].dropna().unique().tolist())
            filters['confianca'] = st.selectbox("📊 Confiança", confianca_opts)
        
        # Previsão consensual
        if 'previsao_consensual_ajustada' in data_manager.data.columns:
            consenso_opts = ['Todos', 'Sim', 'Não']
            filters['consenso'] = st.selectbox("✅ Consenso", consenso_opts)
        
        # Data
        if 'Data' in data_manager.data.columns:
            datas = ['Todas'] + sorted(data_manager.data['Data'].dropna().unique().tolist(), reverse=True)
            filters['data'] = st.selectbox("📅 Data", datas[:10])  # Mostrar apenas as 10 mais recentes
        
        st.markdown("---")
        
        # Botão para limpar filtros
        if st.button("🧹 Limpar Filtros", use_container_width=True):
            st.rerun()
        
        # Informações
        st.markdown("### ℹ️ Legenda")
        
        with st.expander("Ver estados"):
            st.markdown("""
            - **🟢 green**: Previsão correta
            - **🔴 red**: Previsão incorreta  
            - **⚪ pendente**: Aguardando resultado
            """)
        
        with st.expander("Ver confiança"):
            st.markdown("""
            - **🟢 Alta Confiança**: Score ≥ 75
            - **🟡 Média Confiança**: Score 65-74
            - **🔴 Baixa Confiança**: Score 50-64
            - **⚪ Muito Baixa**: Score < 50
            """)
        
        return filters

def apply_filters(data, filters):
    """Aplica os filtros aos dados"""
    if data.empty or not filters:
        return data
    
    filtered = data.copy()
    
    # Aplicar filtros básicos
    if filters.get('torneio') and filters['torneio'] != 'Todos':
        filtered = filtered[filtered['Torneio'] == filters['torneio']]
    
    if filters.get('estado') and filters['estado'] != 'Todos':
        filtered = filtered[filtered['Estado_Previsao_Geral'] == filters['estado']]
    
    if filters.get('confianca') and filters['confianca'] != 'Todos':
        filtered = filtered[filtered['nivel_confianca_ajustado'] == filters['confianca']]
    
    if filters.get('consenso') and filters['consenso'] != 'Todos':
        filtered = filtered[filtered['previsao_consensual_ajustada'] == filters['consenso']]
    
    if filters.get('data') and filters['data'] != 'Todas':
        filtered = filtered[filtered['Data'] == filters['data']]
    
    return filtered

def display_game_card(row):
    """Exibe um card individual para o jogo"""
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Linha 1: Cabeçalho do jogo
        col1, col2, col3 = st.columns([4, 2, 1])
        
        with col1:
            home = row.get('Time_Home', 'N/A')
            away = row.get('Time_Away', 'N/A')
            torneio = row.get('Torneio', '')
            
            st.markdown(f"**{home}** vs **{away}**")
            st.caption(f"🎯 {torneio} • {row.get('Data_Hora', '')}")
        
        with col2:
            # Estado da previsão
            estado = row.get('Estado_Previsao_Geral', 'pendente')
            if estado == 'green':
                st.markdown('<span class="estado-green">✅ GREEN</span>', unsafe_allow_html=True)
            elif estado == 'red':
                st.markdown('<span class="estado-red">❌ RED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="estado-pendente">⏳ PENDENTE</span>', unsafe_allow_html=True)
        
        with col3:
            # Confiança
            confianca = row.get('nivel_confianca_ajustado', '')
            if 'Alta' in confianca:
                st.markdown('<span class="confidence-high">🟢</span>', unsafe_allow_html=True)
            elif 'Média' in confianca:
                st.markdown('<span class="confidence-medium">🟡</span>', unsafe_allow_html=True)
            elif 'Baixa' in confianca:
                st.markdown('<span class="confidence-low">🔴</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="confidence-very-low">⚪</span>', unsafe_allow_html=True)
        
        # Linha 2: Dados do jogo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            placar_ht = row.get('PLACAR_HT', '0-0')
            st.metric("HT", placar_ht, delta=None)
        
        with col2:
            placar_ft = row.get('PLACAR_FT', '0-0')
            st.metric("FT", placar_ft, delta=None)
        
        with col3:
            # Mostrar status pós-previsão se disponível
            tipo_status_depois = row.get('Tipo_Status_depois_previsao', '')
            if tipo_status_depois and str(tipo_status_depois).lower() == 'inprogress':
                minutos = row.get('Minutos_jogo_depois_previsao', row.get('Minutos_jogo', ''))
                if minutos:
                    st.metric("Minutos", minutos, delta=None)
                else:
                    st.metric("Status", "Em Andamento", delta=None)
            else:
                status = row.get('Status_Simples', '')
                if status == 'Finalizado':
                    st.metric("Status", "FT", delta=None)
                else:
                    st.metric("Status", status, delta=None)
        
        with col4:
            consenso = row.get('previsao_consensual_ajustada', 'Não')
            if consenso == 'Sim':
                st.metric("Consenso", "✅", delta=None)
            else:
                st.metric("Consenso", "❌", delta=None)
        
        # Linha 3: Dados ANTES da previsão
        with st.expander("📊 Dados ANTES da previsão", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Previsões Ativas:**")
                previsoes = []
                if row.get('conceito_Mais_0.5_Golos_SegundaParte') == 'Sim':
                    proba = row.get('pred_Mais_0.5_Golos_SegundaParte_proba', '0%')
                    previsoes.append(f"✅ +0.5 ({proba})")
                
                if row.get('conceito_Mais_1.5_Golos_SegundaParte') == 'Sim':
                    proba = row.get('pred_Mais_1.5_Golos_SegundaParte_proba', '0%')
                    previsoes.append(f"✅ +1.5 ({proba})")
                
                if row.get('conceito_Equipa_Perdendo_Marcar_SegundaParte') == 'Sim':
                    proba = row.get('pred_Equipa_Perdendo_Marcar_SegundaParte_proba', '0%')
                    previsoes.append(f"✅ EQP ({proba})")
                
                if previsoes:
                    for p in previsoes:
                        st.markdown(p)
                else:
                    st.markdown("❌ Sem previsões")
            
            with col2:
                st.markdown("**Métricas:**")
                concordancia = row.get('concordancia_ajustada', 'N/A')
                score = row.get('score_confianca_ajustado', 'N/A')
                st.markdown(f"Concordância: {concordancia}/5")
                st.markdown(f"Score: {score}")
            
            with col3:
                st.markdown("**Status Original:**")
                st.markdown(f"Status: {row.get('Status', 'N/A')}")
                st.markdown(f"Tipo: {row.get('Tipo_Status', 'N/A')}")
        
        # Linha 4: Dados DEPOIS da previsão
        with st.expander("📈 Dados DEPOIS da previsão", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Evolução do Placar:**")
                evolucao = row.get('evolução do Placar_depois_previsao', 'N/A')
                st.markdown(evolucao if evolucao != '' else "Sem dados")
            
            with col2:
                st.markdown("**Golos por tempo:**")
                minutos_casa = row.get('minutos Golos_Casa_depois_previsao', '')
                minutos_fora = row.get('minutos Golos_Fora_depois_previsao', '')
                if minutos_casa:
                    st.markdown(f"🏠 Casa: {minutos_casa}")
                if minutos_fora:
                    st.markdown(f"✈️ Fora: {minutos_fora}")
            
            with col3:
                st.markdown("**Análise por threshold:**")
                # Mostrar estado para threshold 46
                estado_46 = row.get('Estado_46', 'pendente')
                golos_46 = row.get('Golos_apos_46', 0)
                if estado_46 != '':
                    st.markdown(f"Threshold 46': {estado_46} ({golos_46} golos)")
        
        # Linha 5: Análise detalhada (se disponível)
        detalhes = row.get('Estado_Previsao_Detalhado', '')
        if detalhes and detalhes != '':
            st.caption(f"📝 {detalhes[:100]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_metrics_summary(data):
    """Exibe resumo de métricas"""
    if data.empty:
        return
    
    st.markdown("### 📊 Resumo Estatístico")
    
    # Criar métricas principais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = len(data)
        st.metric("Total Jogos", total)
    
    with col2:
        if 'Estado_Previsao_Geral' in data.columns:
            green = (data['Estado_Previsao_Geral'] == 'green').sum()
            st.metric("Green", green)
    
    with col3:
        if 'Estado_Previsao_Geral' in data.columns:
            red = (data['Estado_Previsao_Geral'] == 'red').sum()
            st.metric("Red", red)
    
    with col4:
        if 'Estado_Previsao_Geral' in data.columns:
            pendente = (data['Estado_Previsao_Geral'] == 'pendente').sum()
            st.metric("Pendente", pendente)
    
    with col5:
        if 'previsao_consensual_ajustada' in data.columns:
            consenso_sim = (data['previsao_consensual_ajustada'] == 'Sim').sum()
            st.metric("Consenso Sim", consenso_sim)

def display_live_games(data):
    """Exibe jogos em andamento (Live) - APENAS jogos com Tipo_Status_depois_previsao = 'inprogress'"""
    if data.empty:
        st.info("📭 Nenhum jogo live encontrado")
        return
    
    # MODIFICAÇÃO: Filtrar APENAS jogos com Tipo_Status_depois_previsao == "inprogress"
    if 'Tipo_Status_depois_previsao' in data.columns:
        # Filtrar jogos com status "inprogress" (case-insensitive)
        live_mask = data['Tipo_Status_depois_previsao'].fillna('').astype(str).str.lower() == 'inprogress'
        live_data = data[live_mask].copy()
        
        # Log para debug
        st.info(f"Filtro aplicado: {sum(live_mask)} jogos com 'Tipo_Status_depois_previsao' = 'inprogress' de {len(data)} total")
        
        # Mostrar exemplos de status para debug
        if st.checkbox("🔍 Mostrar debug de status"):
            st.write("Valores únicos em Tipo_Status_depois_previsao:", data['Tipo_Status_depois_previsao'].unique()[:10])
            st.write("Contagem de status:", data['Tipo_Status_depois_previsao'].value_counts())
    else:
        # Fallback para lógica antiga se coluna não existir
        st.warning("⚠️ Coluna 'Tipo_Status_depois_previsao' não encontrada. Usando lógica antiga.")
        live_status = ['1ª Parte', '2ª Parte', 'Em Andamento']
        live_data = data[data['Status_Simples'].isin(live_status)]
    
    if live_data.empty:
        st.info("📭 Nenhum jogo em andamento no momento")
        return
    
    # Ordenar por minutos (mais avançado primeiro)
    if 'Minutos_jogo_depois_previsao' in live_data.columns:
        # Extrair minutos numéricos para ordenação
        def extract_minutes(x):
            try:
                if pd.isna(x):
                    return 0
                # Extrair apenas o primeiro número (antes do :)
                parts = str(x).split(':')[0]
                return int(parts) if parts.isdigit() else 0
            except:
                return 0
        
        live_data['_minutos_num'] = live_data['Minutos_jogo_depois_previsao'].apply(extract_minutes)
        live_data = live_data.sort_values('_minutos_num', ascending=False)
    
    # Exibir cards
    st.markdown(f"### 🔥 Jogos em Andamento ({len(live_data)})")
    
    for idx, row in live_data.iterrows():
        display_game_card(row)

def display_finished_games(data):
    """Exibe jogos terminados - TODOS os jogos que NÃO estão com Tipo_Status_depois_previsao = 'inprogress'"""
    if data.empty:
        st.info("📭 Nenhum jogo terminado encontrado")
        return
    
    # MODIFICAÇÃO: Filtrar jogos que NÃO têm Tipo_Status_depois_previsao = "inprogress"
    if 'Tipo_Status_depois_previsao' in data.columns:
        # Filtrar jogos que NÃO estão "inprogress"
        finished_mask = data['Tipo_Status_depois_previsao'].fillna('').astype(str).str.lower() != 'inprogress'
        finished_data = data[finished_mask].copy()
        
        # Log para debug
        st.info(f"Filtro aplicado: {sum(finished_mask)} jogos NÃO 'inprogress' de {len(data)} total")
    else:
        # Fallback para lógica antiga se coluna não existir
        st.warning("⚠️ Coluna 'Tipo_Status_depois_previsao' não encontrada. Usando lógica antiga.")
        finished_data = data[data['Status_Simples'] == 'Finalizado']
    
    if finished_data.empty:
        st.info("📭 Nenhum jogo terminado encontrado")
        return
    
    # Ordenar por data mais recente
    if 'Timestamp' in finished_data.columns:
        finished_data = finished_data.sort_values('Timestamp', ascending=False)
    
    # Exibir cards
    st.markdown(f"### ✅ Jogos Terminados ({len(finished_data)})")
    
    # Agrupar por tipo de status para melhor organização
    if 'Tipo_Status_depois_previsao' in finished_data.columns:
        status_groups = finished_data['Tipo_Status_depois_previsao'].unique()
        
        for status in status_groups:
            if pd.isna(status) or status == '':
                continue
                
            group_data = finished_data[finished_data['Tipo_Status_depois_previsao'] == status]
            
            with st.expander(f"{status} ({len(group_data)} jogos)", expanded=True):
                # Métricas específicas para cada grupo
                if not group_data.empty:
                    green_count = (group_data['Estado_Previsao_Geral'] == 'green').sum()
                    red_count = (group_data['Estado_Previsao_Geral'] == 'red').sum()
                    total_resolved = green_count + red_count
                    
                    if total_resolved > 0:
                        accuracy = (green_count / total_resolved * 100)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Acertos", green_count)
                        with col2:
                            st.metric("Erros", red_count)
                        with col3:
                            st.metric("Precisão", f"{accuracy:.1f}%")
                
                # Mostrar jogos do grupo
                for idx, row in group_data.iterrows():
                    display_game_card(row)
    else:
        # Se não tiver a coluna, mostrar todos juntos
        # Métricas específicas para jogos terminados
        if not finished_data.empty:
            green_count = (finished_data['Estado_Previsao_Geral'] == 'green').sum()
            red_count = (finished_data['Estado_Previsao_Geral'] == 'red').sum()
            total_resolved = green_count + red_count
            
            if total_resolved > 0:
                accuracy = (green_count / total_resolved * 100)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acertos", green_count)
                with col2:
                    st.metric("Erros", red_count)
                with col3:
                    st.metric("Precisão", f"{accuracy:.1f}%")
        
        for idx, row in finished_data.iterrows():
            display_game_card(row)

def display_analytics_tab(data):
    """Exibe a aba de análises"""
    if data.empty:
        st.info("📭 Dados insuficientes para análise")
        return
    
    st.markdown("## 📈 Análises e Estatísticas")
    
    # Abas dentro de análises
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Estatísticas", "🎯 Performance", "📅 Temporal", "🔍 Detalhes"])
    
    with tab1:
        # Estatísticas gerais
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribuição por Torneio")
            if 'Torneio' in data.columns:
                torneio_counts = data['Torneio'].value_counts().head(10)
                fig = px.bar(
                    x=torneio_counts.values,
                    y=torneio_counts.index,
                    orientation='h',
                    title="Top 10 Torneios",
                    labels={'x': 'Número de Jogos', 'y': 'Torneio'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribuição por Confiança")
            if 'nivel_confianca_ajustado' in data.columns:
                conf_counts = data['nivel_confianca_ajustado'].value_counts()
                fig = px.pie(
                    values=conf_counts.values,
                    names=conf_counts.index,
                    title="Distribuição por Nível de Confiança",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance por Confiança")
            if 'nivel_confianca_ajustado' in data.columns and 'Estado_Previsao_Geral' in data.columns:
                # Filtrar apenas jogos resolvidos
                resolved = data[data['Estado_Previsao_Geral'].isin(['green', 'red'])]
                if not resolved.empty:
                    performance = resolved.groupby('nivel_confianca_ajustado').apply(
                        lambda x: (x['Estado_Previsao_Geral'] == 'green').mean() * 100
                    ).reset_index(name='accuracy')
                    fig = px.bar(
                        performance,
                        x='nivel_confianca_ajustado',
                        y='accuracy',
                        title="Precisão por Nível de Confiança",
                        labels={'accuracy': 'Precisão (%)', 'nivel_confianca_ajustado': 'Confiança'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Consenso vs Performance")
            if 'previsao_consensual_ajustada' in data.columns and 'Estado_Previsao_Geral' in data.columns:
                resolved = data[data['Estado_Previsao_Geral'].isin(['green', 'red'])]
                if not resolved.empty:
                    consenso_perf = resolved.groupby('previsao_consensual_ajustada').apply(
                        lambda x: (x['Estado_Previsao_Geral'] == 'green').mean() * 100
                    ).reset_index(name='accuracy')
                    fig = px.bar(
                        consenso_perf,
                        x='previsao_consensual_ajustada',
                        y='accuracy',
                        title="Performance do Consenso",
                        labels={'accuracy': 'Precisão (%)', 'previsao_consensual_ajustada': 'Consenso'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Análise temporal
        if 'Timestamp' in data.columns:
            st.markdown("#### Jogos por Hora do Dia")
            data['Hora'] = pd.to_datetime(data['Timestamp']).dt.hour
            hora_counts = data['Hora'].value_counts().sort_index()
            fig = px.line(
                x=hora_counts.index,
                y=hora_counts.values,
                title="Distribuição por Hora",
                labels={'x': 'Hora do Dia', 'y': 'Número de Jogos'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Tabela detalhada
        st.markdown("#### 📋 Dados Detalhados")
        
        # Colunas para mostrar
        columns_to_show = [
            'ID_Jogo', 'Torneio', 'Time_Home', 'Time_Away',
            'PLACAR_HT', 'PLACAR_FT', 'Status_Simples',
            'Tipo_Status_depois_previsao',  # NOVA COLUNA
            'Estado_Previsao_Geral', 'nivel_confianca_ajustado',
            'previsao_consensual_ajustada', 'Data_Hora'
        ]
        
        existing_cols = [col for col in columns_to_show if col in data.columns]
        
        if existing_cols:
            st.dataframe(
                data[existing_cols].head(50),
                use_container_width=True,
                height=400
            )
            
            # Botão de download
            csv = data[existing_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name=f"analise_previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def display_about_tab():
    """Exibe a aba sobre"""
    st.markdown("## ℹ️ Sobre o Sistema")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Sistema de Previsões SofaScore
        
        Este sistema utiliza modelos de machine learning para prever eventos em jogos de futebol em tempo real.
        
        #### 📊 Conceitos Previstos
        
        1. **+0.5 Golos 2ª Parte**
           - Previsão de mais de 0.5 gols no total da segunda parte
        
        2. **+1.5 Golos 2ª Parte**
           - Previsão de mais de 1.5 gols no total da segunda parte
        
        3. **Equipa Perdendo Marcar (EQP)**
           - Previsão de que a equipa que está perdendo no intervalo marque na segunda parte
        
        #### ⚙️ Fluxo de Dados
        
        1. **Coleta**: Dados em tempo real da SofaScore
        2. **Processamento**: Análise com modelos de ML
        3. **Previsão**: Cálculo de probabilidades
        4. **Validação**: Comparação com resultados reais
        5. **Dashboard**: Visualização interativa
        
        #### 📈 Métricas de Avaliação
        
        - **Estado da Previsão**: GREEN (correta), RED (incorreta), PENDENTE (aguardando)
        - **Nível de Confiança**: Classificação baseada em múltiplos fatores
        - **Consenso**: Concordância entre diferentes métodos de previsão
        - **Score**: Pontuação numérica de confiança (0-100)
        """)
    
    with col2:
        st.markdown("""
        #### 🏆 Estatísticas Atuais
        """)
        
        # Espaço para métricas rápidas
        st.metric("Versão", "2.0.0")
        st.metric("Última Atualização", datetime.now().strftime('%d/%m/%Y'))
        
        st.markdown("""
        #### 📞 Suporte
        
        **Problemas Comuns:**
        1. Dados não carregando
        2. Atualizações atrasadas
        3. Previsões inconsistentes
        
        **Soluções:**
        1. Verificar conexão com dados
        2. Executar scripts de atualização
        3. Recalibrar modelos
        
        #### 🔄 Atualização
        
        O sistema é atualizado automaticamente a cada 5 minutos.
        Para forçar atualização, clique no botão "Atualizar".
        """)

def main():
    """Função principal da aplicação"""
    
    # Inicializar gerenciador de dados
    data_manager = DataManager()
    
    # Header principal
    display_header()
    
    if data_manager.data is None or data_manager.data.empty:
        st.warning("⚠️ Não foi possível carregar os dados. Verifique se o arquivo de previsões foi gerado.")
        
        # Mostrar instruções
        with st.expander("📝 Instruções para gerar dados"):
            st.markdown("""
            1. **Execute o script de previsões:**
            ```bash
            cd /workspaces/previsao_sofascore
            python scripts/carregar_modelos.py
            ```
            
            2. **Execute o reprocessamento:**
            ```bash
            python scripts/reprocessar_golos_com_registro_efetivo_refactor.py
            ```
            
            3. **Execute a análise:**
            ```bash
            python scripts/tarefa7_refactor_usando_depois_previsao.py
            ```
            
            4. **Execute o patch de status:**
            ```bash
            python scripts/patch_status_previsao.py
            ```
            """)
        return
    
    # Verificar se a coluna Tipo_Status_depois_previsao existe
    if 'Tipo_Status_depois_previsao' not in data_manager.data.columns:
        st.warning("⚠️ Coluna 'Tipo_Status_depois_previsao' não encontrada. Execute o patch primeiro.")
        
        with st.expander("📝 Como aplicar o patch"):
            st.markdown("""
            Execute o comando:
            ```bash
            python scripts/patch_status_previsao.py
            ```
            
            Ou use o script:
            ```bash
            ./scripts/patch_rapido.sh
            ```
            
            Este patch irá:
            1. Ler dados de `/workspaces/previsao_sofascore/data/jogos_ativos_depois_previsao.xlsx`
            2. Atualizar `df_previsoes_sim_concatenado.csv` com a coluna `Tipo_Status_depois_previsao`
            3. Criar backup do arquivo original
            """)
        
        # Oferecer para executar o patch
        if st.button("🚀 Executar Patch Agora"):
            import subprocess
            with st.spinner("Executando patch..."):
                result = subprocess.run(
                    ["python", "scripts/patch_status_previsao.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("✅ Patch executado com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao executar patch:")
                    st.code(result.stderr)
    
    # Sidebar com filtros
    with st.sidebar:
        filters = display_sidebar(data_manager)
    
    # Aplicar filtros
    filtered_data = apply_filters(data_manager.data, filters)
    
    # Métricas principais
    display_metrics_summary(filtered_data)
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Live", "✅ Terminados", "📈 Análises", "ℹ️ Sobre"])
    
    with tab1:
        display_live_games(filtered_data)
    
    with tab2:
        display_finished_games(filtered_data)
    
    with tab3:
        display_analytics_tab(filtered_data)
    
    with tab4:
        display_about_tab()

# Ponto de entrada principal
if __name__ == "__main__":
    # Suprimir warnings do Streamlit
    import warnings
    warnings.filterwarnings('ignore')
    
    # Executar aplicação
    main()