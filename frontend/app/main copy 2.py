# ⚽ Frontend Streamlit para Previsões SofaScore
# Versão refatorada com gráficos dinâmicos, timezones e links Bet365

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import re
import unicodedata
import urllib.parse
from zoneinfo import ZoneInfo

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
    
    /* Links Bet365 */
    .bet365-link {
        background: linear-gradient(135deg, #00a335 0%, #00662e 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
        font-size: 0.8rem;
    }
    
    /* Timezone badge */
    .timezone-badge {
        background: #6B7280;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ========== FUNÇÕES AUXILIARES ==========

def clean_team_name_for_url(name: str, to_lower: bool = False, max_len: int | None = None) -> str:
    """Normaliza um nome para uso em URLs"""
    if not name:
        return ""
    # 1) '-' explícito -> espaço
    s = name.replace("-", " ")
    # 2) Normalização unicode (remove acentos)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")  # deixa apenas ASCII
    # 3) Substitui qualquer coisa que não seja letra, dígito ou espaço por espaço
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    # 4) Colapsa espaços múltiplos e remove espaços nas extremidades
    s = re.sub(r"\s+", " ", s).strip()
    # 5) Opcional: minúsculas
    if to_lower:
        s = s.lower()
    # 6) Opcional: limite de comprimento
    if max_len is not None and max_len > 0:
        s = s[:max_len].rstrip()
    return s

def parse_evolucao_placar(evolucao_text):
    """Analisa a evolução do placar para extrair eventos de golos - SEM DUPLICAÇÃO"""
    eventos = []
    placar_final = (0, 0)
    try:
        if pd.isna(evolucao_text) or evolucao_text in ['', 'N/A', 'Jogo Sem Golos']:
            return eventos, placar_final

        texto = str(evolucao_text)
        # Usar dicionário para evitar duplicação por minuto
        eventos_dict = {}

        # Simplificar parsing
        partes = texto.split('→')
        for parte in partes:
            parte = parte.strip()
            if not parte:
                continue

            # Tentar extrair placar e minuto
            padroes = [
                r'(\d+)-(\d+).*?(\d+)',  # 1-0 (25')
                r'(\d+)\s*-\s*(\d+).*?m?(\d+)',  # 1 - 0 aos 25
            ]

            for padrao in padroes:
                match = re.search(padrao, parte)
                if match:
                    try:
                        gc = int(match.group(1))
                        gf = int(match.group(2))
                        minuto = int(match.group(3))
                        if 0 <= minuto <= 130:
                            # Usar minuto como chave para evitar duplicação
                            eventos_dict[minuto] = (gc, gf)
                    except (ValueError, IndexError):
                        continue

        # Converter dicionário para lista
        for minuto, (gc, gf) in eventos_dict.items():
            eventos.append((minuto, gc, gf))

        # Ordenar por minuto
        eventos.sort(key=lambda x: x[0])

        # Obter placar final
        if eventos:
            placar_final = (eventos[-1][1], eventos[-1][2])

        return eventos, placar_final

    except Exception as e:
        print(f"⚠️ Erro ao analisar evolução do placar: {e}")
        return eventos, placar_final

def get_momentum_data(game_id, df_pontos):
    """Obtém dados de momentum para um jogo específico"""
    try:
        if df_pontos is None or df_pontos.empty:
            return []

        # Converter game_id para string para comparação
        game_id_str = str(game_id)

        # Procurar o jogo
        matching_rows = df_pontos[df_pontos['ID_Jogo'].astype(str) == game_id_str]

        if len(matching_rows) == 0:
            return []

        # Pegar a primeira linha correspondente
        row = matching_rows.iloc[0]

        # Extrair os pontos (colunas 1 a 90)
        pontos = []
        for i in range(1, 91):  # 1 a 90
            col_name = str(i)
            if col_name in df_pontos.columns:
                try:
                    valor = row[col_name]
                    if pd.isna(valor):
                        pontos.append(0.0)
                    else:
                        pontos.append(float(valor))
                except:
                    pontos.append(0.0)
            else:
                pontos.append(0.0)

        return pontos

    except Exception as e:
        return []

def create_momentum_chart(game_id, time_home, time_away, momentum_data, eventos_golos, 
                         placar_final, status_jogo, minutos_atual=None):
    """Cria gráfico de momentum interativo com Plotly"""
    
    # Criar minutos (1 a 90)
    minutos = list(range(1, len(momentum_data) + 1))
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar linha de momentum
    fig.add_trace(go.Scatter(
        x=minutos,
        y=momentum_data,
        mode='lines',
        name='Momentum',
        line=dict(color='#3B82F6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Adicionar linha de referência em 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Adicionar linhas verticais para os tempos
    fig.add_vline(x=45, line_dash="dot", line_color="green", opacity=0.7)
    fig.add_vline(x=90, line_dash="dot", line_color="red", opacity=0.7)
    
    # Adicionar marcadores de golos
    for minuto, gc, gf in eventos_golos:
        # Determinar cor baseada em quem marcou
        if gc > 0 and gf == 0:  # Gol da casa
            marker_color = '#2196F3'
            symbol = 'triangle-up'
            size = 15
            name = f'Gol {time_home}'
        elif gf > 0 and gc == 0:  # Gol fora
            marker_color = '#FF5722'
            symbol = 'triangle-down'
            size = 15
            name = f'Gol {time_away}'
        else:  # Gol de ambos no mesmo minuto (raro)
            marker_color = '#FFD700'
            symbol = 'star'
            size = 20
            name = 'Gol ambas'
        
        # Adicionar marcador
        fig.add_trace(go.Scatter(
            x=[minuto],
            y=[momentum_data[minuto-1] if minuto-1 < len(momentum_data) else 0],
            mode='markers',
            name=name,
            marker=dict(
                color=marker_color,
                size=size,
                symbol=symbol,
                line=dict(color='white', width=2)
            ),
            hovertemplate=f'Minuto: {minuto}<br>Placar: {gc}-{gf}<extra></extra>'
        ))
        
        # Adicionar texto do placar
        fig.add_annotation(
            x=minuto,
            y=momentum_data[minuto-1] + 5 if minuto-1 < len(momentum_data) else 5,
            text=f"{gc}-{gf}",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor="white",
            bordercolor=marker_color,
            borderwidth=1,
            borderpad=2,
            opacity=0.9
        )
    
    # Adicionar linha para tempo atual (se jogo em andamento)
    if minutos_atual and str(minutos_atual).lower() != 'terminado':
        try:
            # Extrair minuto atual
            if ':' in str(minutos_atual):
                min_atual = int(str(minutos_atual).split(':')[0])
            else:
                min_atual = int(str(minutos_atual))
            
            if 0 < min_atual <= 90:
                fig.add_vline(
                    x=min_atual, 
                    line_dash="solid", 
                    line_color="orange", 
                    line_width=3,
                    opacity=0.7,
                    annotation_text=f"Minuto {min_atual}"
                )
        except:
            pass
    
    # Configurar layout
    title = f"{time_home} vs {time_away}"
    if placar_final:
        title += f" | Placar Final: {placar_final[0]}-{placar_final[1]}"
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title="Minuto",
            range=[0, 91],
            gridcolor='lightgray',
            dtick=10
        ),
        yaxis=dict(
            title="Momentum",
            gridcolor='lightgray'
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=400
    )
    
    # Adicionar anotação para status
    status_text = f"Status: {status_jogo}"
    if minutos_atual:
        status_text += f" ({minutos_atual})"
    
    fig.add_annotation(
        x=0.01,
        y=1.05,
        xref="paper",
        yref="paper",
        text=status_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bgcolor="lightyellow",
        bordercolor="orange",
        borderwidth=1
    )
    
    return fig

def get_current_timezones():
    """Retorna horário atual em UTC e UTC-1"""
    utc_now = datetime.now(ZoneInfo("UTC"))
    utc_minus_one = utc_now.astimezone(ZoneInfo("Etc/GMT+1"))
    
    return {
        'UTC': utc_now.strftime('%H:%M'),
        'UTC-1': utc_minus_one.strftime('%H:%M')
    }

# ========== CLASSES DE GERENCIAMENTO ==========

class DataManager:
    """Gerencia o carregamento e processamento dos dados"""
    
    def __init__(self):
        # Caminhos relativos
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_path = self.base_dir / "data" / "df_previsoes_sim_concatenado.csv"
        self.pontos_path = self.base_dir / "data" / "dados_pontos_geral.csv"
        self.data = None
        self.df_pontos = None
        self.load_data()
    
    def load_data(self):
        """Carrega os dados dos arquivos CSV"""
        try:
            # Carregar dados de previsões
            if self.data_path.exists():
                self.data = pd.read_csv(self.data_path, dtype=str, low_memory=False)
                self.data = self.clean_data(self.data)
                self.data = self.process_data(self.data)
            else:
                st.error(f"❌ Arquivo não encontrado: {self.data_path}")
                return False
            
            # Carregar dados de momentum
            if self.pontos_path.exists():
                self.df_pontos = pd.read_csv(self.pontos_path, dtype=str, low_memory=False)
                st.success(f"✅ Dados de momentum carregados: {len(self.df_pontos)} jogos")
            else:
                st.warning(f"⚠️ Arquivo de momentum não encontrado: {self.pontos_path}")
                self.df_pontos = pd.DataFrame()
            
            return True
                    
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
        
        # Normalizar Tipo_Status_depois_previsao
        if 'Tipo_Status_depois_previsao' in df.columns:
            df['Tipo_Status_depois_previsao'] = df['Tipo_Status_depois_previsao'].astype(str).str.lower()
        else:
            df['Tipo_Status_depois_previsao'] = 'unknown'
        
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

# ========== FUNÇÕES DE VISUALIZAÇÃO ==========

def display_header(data_manager):
    """Exibe o cabeçalho da aplicação"""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
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
        # Obter horários em diferentes timezones
        timezones = get_current_timezones()
        st.markdown(f"""
        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px;">
            <div style="font-size: 0.9rem; color: #666;">Horários Atuais:</div>
            <div class="timezone-badge">UTC: {timezones['UTC']}</div>
            <div class="timezone-badge">UTC-1: {timezones['UTC-1']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.write("")
        st.write("")
        if data_manager.data is not None:
            total_jogos = len(data_manager.data)
            st.metric("Total Jogos", total_jogos)

def display_sidebar(data_manager):
    """Cria a sidebar com filtros e informações"""
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        # Informações do sistema
        st.markdown("### 📊 Estatísticas")
        if data_manager.data is not None and not data_manager.data.empty:
            total_jogos = len(data_manager.data)
            
            # Contar jogos por status
            if 'Tipo_Status_depois_previsao' in data_manager.data.columns:
                live_jogos = len(data_manager.data[
                    data_manager.data['Tipo_Status_depois_previsao'] == 'inprogress'
                ])
                finalizados = len(data_manager.data[
                    ~data_manager.data['Tipo_Status_depois_previsao'].isin(['inprogress', 'unknown', 'nan'])
                ])
                outros = total_jogos - live_jogos - finalizados
            else:
                live_jogos = 0
                finalizados = 0
                outros = total_jogos
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", total_jogos)
            with col2:
                st.metric("Live", live_jogos)
            with col3:
                st.metric("Finalizados", finalizados)
        
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
        
        # Filtro por status pós-previsão
        if 'Tipo_Status_depois_previsao' in data_manager.data.columns:
            status_opts = ['Todos'] + sorted(data_manager.data['Tipo_Status_depois_previsao'].dropna().unique().tolist())
            filters['tipo_status'] = st.selectbox("📊 Status Pós-Previsão", status_opts)
        
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
            filters['data'] = st.selectbox("📅 Data", datas[:10])
        
        st.markdown("---")
        
        # Links rápidos
        st.markdown("### 🔗 Links Rápidos")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Ver Gráficos", use_container_width=True):
                st.session_state['show_charts'] = not st.session_state.get('show_charts', False)
                st.rerun()
        
        with col2:
            if st.button("📥 Exportar CSV", use_container_width=True):
                st.session_state['export_data'] = True
        
        st.markdown("---")
        
        # Botão para limpar filtros
        if st.button("🧹 Limpar Filtros", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith('filter_'):
                    del st.session_state[key]
            st.rerun()
        
        return filters

def apply_filters(data, filters):
    """Aplica os filtros aos dados"""
    if data.empty or not filters:
        return data
    
    filtered = data.copy()
    
    # Aplicar filtros básicos
    if filters.get('torneio') and filters['torneio'] != 'Todos':
        filtered = filtered[filtered['Torneio'] == filters['torneio']]
    
    if filters.get('tipo_status') and filters['tipo_status'] != 'Todos':
        filtered = filtered[filtered['Tipo_Status_depois_previsao'] == filters['tipo_status']]
    
    if filters.get('estado') and filters['estado'] != 'Todos':
        filtered = filtered[filtered['Estado_Previsao_Geral'] == filters['estado']]
    
    if filters.get('confianca') and filters['confianca'] != 'Todos':
        filtered = filtered[filtered['nivel_confianca_ajustado'] == filters['confianca']]
    
    if filters.get('consenso') and filters['consenso'] != 'Todos':
        filtered = filtered[filtered['previsao_consensual_ajustada'] == filters['consenso']]
    
    if filters.get('data') and filters['data'] != 'Todas':
        filtered = filtered[filtered['Data'] == filters['data']]
    
    return filtered

def display_game_card(row, data_manager, show_chart=True):
    """Exibe um card individual para o jogo com gráfico de momentum"""
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
            placar_depois = row.get('Placar_depois_previsao', placar_ft)
            st.metric("FT", placar_depois, delta=None)
        
        with col3:
            tipo_status = row.get('Tipo_Status_depois_previsao', '')
            minutos = row.get('Minutos_jogo_depois_previsao', row.get('Minutos_jogo', ''))
            
            if tipo_status == 'inprogress':
                st.metric("Status", "LIVE", delta=None)
                if minutos:
                    st.caption(f"Minuto: {minutos}")
            elif tipo_status in ['ended', 'finished']:
                st.metric("Status", "TERMINADO", delta=None)
            else:
                st.metric("Status", tipo_status.upper(), delta=None)
        
        with col4:
            consenso = row.get('previsao_consensual_ajustada', 'Não')
            if consenso == 'Sim':
                st.metric("Consenso", "✅", delta=None)
            else:
                st.metric("Consenso", "❌", delta=None)
        
        # Linha 3: Links Bet365
        time_home = row.get('Time_Home', '')
        time_away = row.get('Time_Away', '')
        
        if time_home not in ['', 'N/A', '0']:
            time_home_clean = clean_team_name_for_url(time_home, to_lower=False)
            time_home_encoded = urllib.parse.quote(time_home_clean)
            
            st.markdown(f"""
            <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <div style="font-size: 0.9rem; margin-bottom: 5px;">🔗 Links Bet365:</div>
                <a href="https://www.bet365.com/#/AX/K%5E{time_home_encoded}" target="_blank" class="bet365-link">
                    {time_home}
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        # Linha 4: Gráfico de Momentum (se habilitado)
        if show_chart and data_manager.df_pontos is not None and not data_manager.df_pontos.empty:
            game_id = str(row.get('ID_Jogo', ''))
            
            # Obter dados de momentum
            momentum_data = get_momentum_data(game_id, data_manager.df_pontos)
            
            # Obter eventos de golos
            evolucao_text = row.get('evolução do Placar_depois_previsao', '')
            eventos_golos, placar_final = parse_evolucao_placar(evolucao_text)
            
            # Criar gráfico se houver dados
            if momentum_data:
                minutos_atual = row.get('Minutos_jogo_depois_previsao', row.get('Minutos_jogo', ''))
                tipo_status = row.get('Tipo_Status_depois_previsao', '')
                
                fig = create_momentum_chart(
                    game_id=game_id,
                    time_home=time_home,
                    time_away=time_away,
                    momentum_data=momentum_data,
                    eventos_golos=eventos_golos,
                    placar_final=placar_final,
                    status_jogo=tipo_status,
                    minutos_atual=minutos_atual
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Linha 5: Dados ANTES da previsão
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
        
        # Linha 6: Dados DEPOIS da previsão
        with st.expander("📈 Dados DEPOIS da previsão", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Evolução do Placar:**")
                evolucao = row.get('evolução do Placar_depois_previsao', 'N/A')
                if evolucao and evolucao != 'N/A':
                    # Mostrar eventos de golos formatados
                    eventos_golos, _ = parse_evolucao_placar(evolucao)
                    if eventos_golos:
                        for minuto, gc, gf in eventos_golos:
                            st.markdown(f"⚽ {minuto}': {gc}-{gf}")
                    else:
                        st.markdown("Sem golos registrados")
                else:
                    st.markdown("Sem dados")
            
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
                estado_46 = row.get('Estado_46', 'pendente')
                golos_46 = row.get('Golos_apos_46', 0)
                if estado_46 != '':
                    st.markdown(f"Threshold 46': {estado_46} ({golos_46} golos)")
        
        # Linha 7: Análise detalhada (se disponível)
        detalhes = row.get('Estado_Previsao_Detalhado', '')
        if detalhes and detalhes != '':
            st.caption(f"📝 {detalhes[:100]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_live_games(data, data_manager):
    """Exibe jogos em andamento (Live) - APENAS jogos com Tipo_Status_depois_previsao = 'inprogress'"""
    if data.empty:
        st.info("📭 Nenhum jogo live encontrado")
        return
    
    # Filtrar jogos com status "inprogress"
    live_mask = data['Tipo_Status_depois_previsao'] == 'inprogress'
    live_data = data[live_mask].copy()
    
    if live_data.empty:
        st.info("📭 Nenhum jogo em andamento no momento")
        
        # Mostrar estatísticas de status para debug
        with st.expander("🔍 Debug - Distribuição de Status"):
            status_counts = data['Tipo_Status_depois_previsao'].value_counts()
            st.dataframe(status_counts)
            
            # Mostrar valores únicos
            st.write("Valores únicos em Tipo_Status_depois_previsao:")
            st.write(data['Tipo_Status_depois_previsao'].unique()[:20])
        return
    
    # Ordenar por minutos (mais avançado primeiro)
    if 'Minutos_jogo_depois_previsao' in live_data.columns:
        # Extrair minutos numéricos para ordenação
        def extract_minutes(x):
            try:
                if pd.isna(x):
                    return 0
                if str(x).lower() == 'terminado':
                    return 999
                # Extrair apenas o primeiro número (antes do :)
                parts = str(x).split(':')[0]
                return int(parts) if parts.isdigit() else 0
            except:
                return 0
        
        live_data['_minutos_num'] = live_data['Minutos_jogo_depois_previsao'].apply(extract_minutes)
        live_data = live_data.sort_values('_minutos_num', ascending=False)
    
    # Exibir cards
    st.markdown(f"### 🔥 Jogos em Andamento ({len(live_data)})")
    
    show_charts = st.session_state.get('show_charts', True)
    
    for idx, row in live_data.iterrows():
        display_game_card(row, data_manager, show_chart=show_charts)

def display_finished_games(data, data_manager):
    """Exibe jogos terminados - TODOS os jogos que NÃO estão com Tipo_Status_depois_previsao = 'inprogress'"""
    if data.empty:
        st.info("📭 Nenhum jogo terminado encontrado")
        return
    
    # Filtrar jogos que NÃO têm Tipo_Status_depois_previsao = "inprogress"
    finished_mask = data['Tipo_Status_depois_previsao'] != 'inprogress'
    finished_data = data[finished_mask].copy()
    
    # Remover valores nulos/desconhecidos se quiser
    finished_data = finished_data[~finished_data['Tipo_Status_depois_previsao'].isin(['unknown', 'nan', ''])]
    
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
            
            with st.expander(f"{status.upper()} ({len(group_data)} jogos)", expanded=True):
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
                show_charts = st.session_state.get('show_charts', True)
                for idx, row in group_data.iterrows():
                    display_game_card(row, data_manager, show_chart=show_charts)
    else:
        # Se não tiver a coluna, mostrar todos juntos
        show_charts = st.session_state.get('show_charts', True)
        for idx, row in finished_data.iterrows():
            display_game_card(row, data_manager, show_chart=show_charts)

def display_analytics_tab(data, data_manager):
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
            st.markdown("#### Distribuição por Status")
            if 'Tipo_Status_depois_previsao' in data.columns:
                status_counts = data['Tipo_Status_depois_previsao'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Distribuição por Status",
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
            'PLACAR_HT', 'PLACAR_FT', 'Placar_depois_previsao',
            'Tipo_Status_depois_previsao', 'Minutos_jogo_depois_previsao',
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

def main():
    """Função principal da aplicação"""
    
    # Inicializar session state
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    
    # Inicializar gerenciador de dados
    data_manager = DataManager()
    
    # Header principal
    display_header(data_manager)
    
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
        return
    
    # Sidebar com filtros
    with st.sidebar:
        filters = display_sidebar(data_manager)
    
    # Aplicar filtros
    filtered_data = apply_filters(data_manager.data, filters)
    
    # Controles de visualização
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 Jogos Filtrados: {len(filtered_data)}")
    
    with col2:
        show_charts = st.checkbox("📈 Mostrar Gráficos", value=st.session_state.show_charts)
        st.session_state.show_charts = show_charts
    
    with col3:
        if st.button("🔄 Atualizar Dados", use_container_width=True):
            data_manager.load_data()
            st.rerun()
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Live", "✅ Terminados", "📈 Análises", "ℹ️ Sobre"])
    
    with tab1:
        display_live_games(filtered_data, data_manager)
    
    with tab2:
        display_finished_games(filtered_data, data_manager)
    
    with tab3:
        display_analytics_tab(filtered_data, data_manager)
    
    with tab4:
        # Aba sobre simplificada
        st.markdown("## ℹ️ Sobre o Sistema")
        st.markdown("""
        ### 🎯 Sistema de Previsões SofaScore
        
        Este sistema utiliza modelos de machine learning para prever eventos em jogos de futebol em tempo real.
        
        #### 📊 Características:
        
        1. **Previsões em Tempo Real**: Análise durante os jogos
        2. **Gráficos de Momentum**: Visualização interativa da dinâmica do jogo
        3. **Status Atualizado**: Informações pós-previsão em tempo real
        4. **Links Bet365**: Acesso rápido às odds
        5. **Múltiplos Fusos Horários**: UTC e UTC-1
        
        #### ⚙️ Fluxo de Dados:
        
        1. **Coleta**: Dados em tempo real da SofaScore
        2. **Processamento**: Análise com modelos de ML
        3. **Previsão**: Cálculo de probabilidades
        4. **Validação**: Comparação com resultados reais
        5. **Dashboard**: Visualização interativa
        
        #### 📞 Suporte:
        
        Para problemas ou sugestões, verifique a documentação ou contate o administrador.
        """)

# Ponto de entrada principal
if __name__ == "__main__":
    # Suprimir warnings do Streamlit
    import warnings
    warnings.filterwarnings('ignore')
    
    # Executar aplicação
    main()