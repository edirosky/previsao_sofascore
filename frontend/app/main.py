# ⚽ Frontend Streamlit para Previsões SofaScore - VERSÃO COMPLETA COM GRÁFICOS MATPLOTLIB E SCHEDULER
# Mantém TODAS as funcionalidades: filtros, análises, métricas, etc.
# Adiciona gráficos matplotlib para momentum com suporte a tempo real
# Inclui scheduler interno para atualização automática

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
import warnings
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para produção
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess
import time
import threading
import queue
import json

# Configuração da página - DEVE SER A PRIMEIRA COISA
st.set_page_config(
    page_title="Previsões SofaScore",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar caminho para importações
sys.path.append(str(Path(__file__).parent.parent))

# Configurações
CONFIG_FILE = Path("/workspaces/previsao_sofascore/config.json")
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except:
        config = {"admin_password": "admin123"}
else:
    config = {"admin_password": "admin123"}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

ADMIN_PASSWORD = config.get("admin_password", "admin123")

# CSS customizado corrigido
st.markdown("""
<style>
    /* Reset para evitar sobreposição */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        z-index: 100;
    }
    
    /* Container para métricas horizontais */
    .metrics-horizontal-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Métricas horizontais */
    .horizontal-metric-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .horizontal-metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 0.3rem;
        line-height: 1;
    }
    
    .horizontal-metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 50;
        height: 100%;
    }
    
    /* Cards compactos de métricas */
    .compact-metric {
        background: white;
        padding: 0.6rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 100%;
    }
    
    .compact-metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 0.2rem;
    }
    
    .compact-metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Cards de jogos */
    .game-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        position: relative;
        z-index: 50;
        overflow: visible !important;
    }
    
    /* Container para gráficos matplotlib */
    .matplotlib-container {
        background: #E0F7FA !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #B2EBF2;
    }
    
    /* Container para scheduler */
    .scheduler-container {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%) !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #667eea;
    }
    
    /* Barra de progresso customizada */
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        width: 0%;
        border-radius: 10px;
        transition: width 0.3s ease;
        text-align: center;
        color: white;
        line-height: 20px;
        font-weight: bold;
        font-size: 12px;
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
    
    /* Badge de timezone */
    .timezone-badge {
        background: #4B5563;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
    }
    
    /* Link Bet365 */
    .bet365-link {
        background: linear-gradient(135deg, #00a335 0%, #00662e 100%);
        color: white !important;
        padding: 6px 12px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
        font-size: 0.85rem;
    }
    
    .bet365-link:hover {
        opacity: 0.9;
        color: white !important;
        text-decoration: none;
    }
    
    /* Melhorias para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Garantir que gráficos não sobreponham */
    .js-plotly-plot {
        position: relative;
        z-index: 10 !important;
    }
    
    /* Estilo para info do jogo compacta */
    .game-info-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Título das métricas */
    .metrics-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== CLASSE SCHEDULER ==========

class SchedulerService:
    """Serviço de scheduler para executar scripts periodicamente"""
    
    def __init__(self, interval_minutes=1):
        self.interval_minutes = interval_minutes
        self.is_running = False
        self.current_cycle = 0
        self.last_run = None
        self.next_run = None
        self.status_queue = queue.Queue()
        self.progress = 0
        self.current_script = ""
        
        # Definir caminho correto para scripts
        self.scripts_dir = Path("/workspaces/previsao_sofascore/scripts")
        self.data_dir = Path("/workspaces/previsao_sofascore/data")
        
        # Scripts com seus nomes corretos
        self.scripts = [
            ("010_collect_and_process.py", "Coleta de eventos", 300),
            ("011_reprocess_goals.py", "Reprocessamento de gols", 300),
            ("012_criar_incidentes_estatisticas_geral.py", "Criação de incidentes", 300),
            ("020_carregar_modelos.py", "Carregamento de modelos", 300),
            ("021_collect_and_process_depois_previsao.py", "Processamento pós-previsão", 300),
            ("030_reprocessar_golos_com_registro_efetivo_refactor.py", "Reprocessamento com registro", 300),
            ("031_patch_simples_tipo_status.py", "Patch de status", 300),
            ("040_analise_metricas_confianca.py", "Análise de métricas", 300),
            ("041_dados_pontos_geral.py", "Dados de pontos", 300)
        ]
        
        self.total_scripts = len(self.scripts)
        self.thread = None
    
    def run_script(self, script_name, description, timeout):
        """Executa um script Python"""
        try:
            self.current_script = description
            self.status_queue.put(f"▶️ Executando: {description}")
            
            script_path = self.scripts_dir / script_name
            
            if not script_path.exists():
                self.status_queue.put(f"❌ Script não encontrado: {script_name} em {script_path}")
                return False
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.scripts_dir  # Executar no diretório correto
            )
            
            if result.returncode == 0:
                self.status_queue.put(f"✅ {description} concluído")
                return True
            else:
                error_msg = result.stderr[:100] if result.stderr else "Sem mensagem de erro"
                self.status_queue.put(f"❌ {description} falhou: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            self.status_queue.put(f"⏰ {description} timeout após {timeout}s")
            return False
        except Exception as e:
            self.status_queue.put(f"⚠️ Erro em {description}: {str(e)[:100]}")
            return False
    
    def run_cycle(self):
        """Executa um ciclo completo de scripts"""
        self.current_cycle += 1
        self.last_run = datetime.now()
        self.next_run = self.last_run + timedelta(minutes=self.interval_minutes)
        
        self.status_queue.put(f"🔄 Iniciando ciclo #{self.current_cycle}")
        self.status_queue.put(f"⏰ Hora de início: {self.last_run.strftime('%H:%M:%S')}")
        
        success_count = 0
        script_index = 0
        
        for script_name, description, timeout in self.scripts:
            script_index += 1
            self.progress = (script_index / self.total_scripts) * 100
            
            # Atualizar progresso
            self.status_queue.put(f"📊 PROGRESS:{self.progress}")
            
            # Executar script
            if self.run_script(script_name, description, timeout):
                success_count += 1
            
            # Pequena pausa entre scripts
            time.sleep(2)
        
        # Aguardar estabilização
        self.status_queue.put("⏳ Aguardando estabilização...")
        time.sleep(5)
        
        self.progress = 100
        self.status_queue.put(f"📊 PROGRESS:{self.progress}")
        
        success_rate = (success_count / self.total_scripts) * 100
        self.status_queue.put(f"🎯 Ciclo #{self.current_cycle} concluído: {success_count}/{self.total_scripts} scripts ({success_rate:.1f}%)")
        self.status_queue.put(f"⏰ Próximo ciclo: {self.next_run.strftime('%H:%M:%S')}")
        
        return success_count
    
    def scheduler_loop(self):
        """Loop principal do scheduler"""
        while self.is_running:
            try:
                self.run_cycle()
                
                # Calcular tempo para próximo ciclo
                wait_seconds = self.interval_minutes * 60
                for i in range(wait_seconds):
                    if not self.is_running:
                        break
                    
                    # Atualizar tempo restante
                    remaining = wait_seconds - i
                    if remaining % 10 == 0:  # Atualizar a cada 10 segundos
                        self.status_queue.put(f"⏳ Aguardando próximo ciclo: {remaining}s restantes")
                    
                    time.sleep(1)
                    
            except Exception as e:
                self.status_queue.put(f"💥 Erro no scheduler: {str(e)}")
                time.sleep(10)  # Espera antes de tentar novamente
    
    def start(self):
        """Inicia o scheduler em uma thread separada"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.scheduler_loop, daemon=True)
            self.thread.start()
            self.status_queue.put("🚀 Scheduler iniciado")
    
    def stop(self):
        """Para o scheduler"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.status_queue.put("🛑 Scheduler parado")
    
    def get_status(self):
        """Obtém o status atual do scheduler"""
        status_messages = []
        while not self.status_queue.empty():
            try:
                message = self.status_queue.get_nowait()
                status_messages.append(message)
            except:
                break
        
        return status_messages

# ========== FUNÇÕES DE GRÁFICOS MATPLOTLIB ==========

def parse_evolucao_placar(evolucao_text):
    """Analisa a evolução do placar para extrair eventos de golos - SEM DUPLICAÇÃO"""
    eventos = []
    placar_final = (0, 0)
    
    try:
        if pd.isna(evolucao_text) or evolucao_text in ['', 'N/A', 'Jogo Sem Golos']:
            return eventos, placar_final

        texto = str(evolucao_text)
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
                            eventos_dict[minuto] = (gc, gf)
                    except:
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
        return eventos, placar_final

def get_momentum_data(game_id, df_pontos):
    """Obtém dados de momentum para um jogo específico"""
    try:
        if df_pontos is None or df_pontos.empty:
            return []

        game_id_str = str(game_id)
        
        # Verificar se a coluna ID_Jogo existe
        if 'ID_Jogo' in df_pontos.columns:
            matching_rows = df_pontos[df_pontos['ID_Jogo'].astype(str) == game_id_str]
        elif 'ID' in df_pontos.columns:
            matching_rows = df_pontos[df_pontos['ID'].astype(str) == game_id_str]
        else:
            # Tentar encontrar por primeiro índice
            id_column = df_pontos.columns[0]
            matching_rows = df_pontos[df_pontos[id_column].astype(str) == game_id_str]

        if len(matching_rows) == 0:
            return []

        row = matching_rows.iloc[0]
        pontos = []
        
        for i in range(1, 91):
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

def create_matplotlib_momentum_chart(game_id, time_home, time_away, momentum_data, 
                                   eventos_golos, placar_final, status_jogo, 
                                   minutos_atual=None, previsao_text=""):
    """Cria gráfico de momentum com matplotlib e retorna como imagem base64"""
    
    try:
        # Criar figura
        fig = plt.figure(figsize=(12, 5), facecolor='#E0F7FA')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#E0F7FA')
        
        # Se não houver dados de momentum, criar gráfico simples
        if not momentum_data or len(momentum_data) == 0:
            momentum_data = [0] * 90
        
        x = np.arange(1, len(momentum_data) + 1)
        colors = ['#4CAF50' if y >= 0 else '#F44336' for y in momentum_data]
        
        # Criar gráfico de barras
        bars = ax.bar(x, momentum_data, width=0.9, color=colors, alpha=0.8)
        
        # Configurar limites do eixo Y
        max_val = max(max(momentum_data) if momentum_data else 0, 
                     abs(min(momentum_data) if momentum_data else 0), 10)
        y_lim = max(max_val * 1.3, 15)
        ax.set_ylim(-y_lim, y_lim)
        
        # Adicionar valores nos bares mais altos
        for xi, yi, bar in zip(x, momentum_data, bars):
            if not np.isnan(yi) and abs(yi) >= max_val * 0.3:
                va_pos = 'bottom' if yi >= 0 else 'top'
                ax.text(xi, yi, f"{yi:.1f}", ha="center", va=va_pos,
                       fontsize=7, fontweight='bold', color='black')
        
        # Configurar eixos e grade
        ax.set_ylabel("Momentum", fontsize=9, fontweight='bold')
        ax.set_xlabel("Minuto", fontsize=9, fontweight='bold')
        ax.set_xticks(np.arange(0, len(momentum_data) + 1, 10))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # Marcar intervalos importantes
        for m in [45, 90]:
            ax.axvline(x=m, linestyle='--', lw=2.0, alpha=0.9, color='#333333')
            ax.text(m, y_lim * 0.92, f"{m}'", ha='center', va='top', fontsize=9,
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
        
        # Adicionar eventos de golos
        for minuto, gc, gf in eventos_golos:
            if minuto > len(momentum_data):
                continue
            
            try:
                idx = minuto - 1
                valor_momentum = momentum_data[idx] if idx < len(momentum_data) else 0
            except Exception:
                valor_momentum = 0
            
            # Cores diferentes para casa/fora
            if gc > 0 and gf == 0:
                marker_color = '#2196F3'  # Azul para gol da casa
                edge_color = '#0D47A1'
            elif gf > 0 and gc == 0:
                marker_color = '#FF5722'  # Laranja para gol fora
                edge_color = '#BF360C'
            else:
                marker_color = '#FFD700'  # Amarelo para gol de ambos
                edge_color = '#FF6F00'
            
            # Posicionar o ícone do golo
            posicao_y_icone = valor_momentum
            
            # Marcador de golo
            ax.scatter(minuto, posicao_y_icone,
                      marker='*',
                      s=200,
                      zorder=10,
                      color=marker_color,
                      edgecolors=edge_color,
                      linewidth=2.0,
                      alpha=0.9)
            
            # Posicionar o texto do placar
            distancia_texto = y_lim * 0.08
            
            if valor_momentum >= 0:
                texto_y = posicao_y_icone + distancia_texto
                vertical_alignment = 'bottom'
            else:
                texto_y = posicao_y_icone - distancia_texto
                vertical_alignment = 'top'
            
            # Texto do placar
            ax.text(minuto, texto_y,
                   f"{gc}-{gf}",
                   ha='center',
                   va=vertical_alignment,
                   fontsize=8,
                   fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.95, 
                            edgecolor=edge_color, linewidth=1.2))
        
        # Adicionar título
        status_text = "TERMINADO" if status_jogo in ['ended', 'finished', 'ft'] else f"LIVE {minutos_atual or ''}"
        
        plt.title(f"{time_home} vs {time_away}\nPlacar: {placar_final[0]}-{placar_final[1]} | Status: {status_text}",
                 fontsize=10, pad=10, fontweight='bold')
        
        # Adicionar previsão na parte inferior se disponível
        if previsao_text:
            if isinstance(previsao_text, str) and len(previsao_text) > 50:
                previsao_text = previsao_text[:50] + "..."
            
            plt.figtext(0.5, 0.01, f"Previsões: {previsao_text}", ha="center",
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Converter figura para base64 para exibição no Streamlit
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, 
                   facecolor=fig.get_facecolor())
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_base64}" class="matplotlib-graph"/>'
        
    except Exception as e:
        plt.close('all')
        return None

# ========== FUNÇÕES AUXILIARES ==========

def clean_team_name_for_url(name: str, to_lower: bool = False, max_len: int | None = None) -> str:
    """Normaliza um nome para uso em URLs"""
    if not name or name in ['', 'N/A', '0']:
        return ""
    
    # 1) '-' explícito -> espaço
    s = name.replace("-", " ")
    
    # 2) Normalização unicode (remove acentos)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    
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

def get_current_timezones():
    """Retorna horário atual em UTC e UTC-1"""
    try:
        utc_now = datetime.now(ZoneInfo("UTC"))
        utc_minus_one = utc_now.astimezone(ZoneInfo("Etc/GMT+1"))
        
        return {
            'UTC': utc_now.strftime('%H:%M'),
            'UTC-1': utc_minus_one.strftime('%H:%M')
        }
    except:
        now = datetime.now()
        return {
            'UTC': now.strftime('%H:%M'),
            'UTC-1': (now - timedelta(hours=1)).strftime('%H:%M')
        }

# ========== CLASSES DE GERENCIAMENTO ==========

class DataManager:
    """Gerencia o carregamento e processamento dos dados"""
    
    def __init__(self):
        self.base_dir = Path("/workspaces/previsao_sofascore")
        self.data_path = self.base_dir / "data" / "df_previsoes_sim_concatenado.csv"
        self.pontos_path = self.base_dir / "data" / "dados_pontos_geral.csv"
        self.data = None
        self.df_pontos = None
        self.load_data()
    
    def load_data(self):
        """Carrega os dados dos arquivos CSV"""
        try:
            if self.data_path.exists():
                self.data = pd.read_csv(self.data_path, dtype=str, low_memory=False)
                self.data = self.clean_data(self.data)
                self.data = self.process_data(self.data)
            else:
                st.error(f"❌ Arquivo não encontrado: {self.data_path}")
                return False
            
            if self.pontos_path.exists():
                self.df_pontos = pd.read_csv(self.pontos_path, dtype=str, low_memory=False)
            else:
                self.df_pontos = pd.DataFrame()
            
            return True
                    
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def clean_data(self, df):
        """Limpa os dados básicos"""
        if df.empty:
            return df
        
        df = df.replace(['', 'nan', 'NaN', 'None', 'null', 'NA'], np.nan)
        
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

def display_horizontal_metrics(data):
    """Exibe métricas na horizontal com design moderno"""
    if data.empty:
        st.info("📭 Sem dados para exibir métricas")
        return
    
    st.markdown('<div class="metrics-horizontal-container">', unsafe_allow_html=True)
    
    # Título
    st.markdown('<div class="metrics-title">📊 RESUMO ESTATÍSTICO</div>', unsafe_allow_html=True)
    
    # Criar 6 colunas (TOTAL + 5 métricas)
    cols = st.columns(6)
    
    # Total de Jogos
    with cols[0]:
        total = len(data)
        st.markdown(f"""
        <div class="horizontal-metric-item">
            <div class="horizontal-metric-value">{total}</div>
            <div class="horizontal-metric-label">TOTAL JOGOS</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Green
    with cols[1]:
        if 'Estado_Previsao_Geral' in data.columns:
            green = (data['Estado_Previsao_Geral'] == 'green').sum()
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value" style="color: #10B981;">{green}</div>
                <div class="horizontal-metric-label">GREEN</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value">0</div>
                <div class="horizontal-metric-label">GREEN</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Red
    with cols[2]:
        if 'Estado_Previsao_Geral' in data.columns:
            red = (data['Estado_Previsao_Geral'] == 'red').sum()
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value" style="color: #EF4444;">{red}</div>
                <div class="horizontal-metric-label">RED</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value">0</div>
                <div class="horizontal-metric-label">RED</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Pendente
    with cols[3]:
        if 'Estado_Previsao_Geral' in data.columns:
            pendente = (data['Estado_Previsao_Geral'] == 'pendente').sum()
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value" style="color: #6B7280;">{pendente}</div>
                <div class="horizontal-metric-label">PENDENTE</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value">0</div>
                <div class="horizontal-metric-label">PENDENTE</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Consenso Sim
    with cols[4]:
        if 'previsao_consensual_ajustada' in data.columns:
            consenso_sim = (data['previsao_consensual_ajustada'] == 'Sim').sum()
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value" style="color: #3B82F6;">{consenso_sim}</div>
                <div class="horizontal-metric-label">CONSENSO SIM</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value">0</div>
                <div class="horizontal-metric-label">CONSENSO SIM</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Live (adicional)
    with cols[5]:
        if 'Tipo_Status_depois_previsao' in data.columns:
            live = (data['Tipo_Status_depois_previsao'] == 'inprogress').sum()
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value" style="color: #F59E0B;">{live}</div>
                <div class="horizontal-metric-label">LIVE AGORA</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="horizontal-metric-item">
                <div class="horizontal-metric-value">0</div>
                <div class="horizontal-metric-label">LIVE AGORA</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_scheduler_status(admin_mode=False):
    """Exibe o status do scheduler (apenas para admin)"""
    if not admin_mode:
        return
    
    st.markdown('<div class="scheduler-container">', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Scheduler de Atualização (Admin)")
    
    # Inicializar scheduler no session_state
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = SchedulerService(interval_minutes=1)
        st.session_state.scheduler.start()
    
    scheduler = st.session_state.scheduler
    
    # Controles do scheduler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if scheduler.is_running:
            st.markdown("**Status:** 🟢 **ATIVO**")
        else:
            st.markdown("**Status:** 🔴 **INATIVO**")
    
    with col2:
        st.markdown(f"**Ciclo atual:** #{scheduler.current_cycle}")
    
    with col3:
        if scheduler.last_run:
            st.markdown(f"**Última execução:** {scheduler.last_run.strftime('%H:%M:%S')}")
    
    # Barra de progresso
    st.markdown(f"**Progresso do ciclo atual:**")
    
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {scheduler.progress}%">
            {scheduler.progress:.1f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    if scheduler.current_script:
        st.markdown(f"**Executando:** {scheduler.current_script}")
    
    # Status messages
    st.markdown("**Logs:**")
    status_messages = scheduler.get_status()
    
    status_container = st.container()
    with status_container:
        for message in status_messages[-10:]:  # Mostrar apenas as últimas 10 mensagens
            if message.startswith("📊 PROGRESS:"):
                continue  # Ignorar mensagens de progresso (já tratadas)
            st.text(message)
    
    # Controles (apenas admin)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Forçar Atualização", use_container_width=True, key="force_update_admin"):
            scheduler.status_queue.put("🔄 Ciclo forçado solicitado...")
            # Executar ciclo em thread separada
            threading.Thread(target=scheduler.run_cycle, daemon=True).start()
    
    with col2:
        if scheduler.is_running:
            if st.button("⏸️ Pausar Scheduler", use_container_width=True, key="pause_scheduler"):
                scheduler.stop()
                st.rerun()
        else:
            if st.button("▶️ Iniciar Scheduler", use_container_width=True, key="start_scheduler"):
                scheduler.start()
                st.rerun()
    
    with col3:
        if st.button("🗑️ Limpar Logs", use_container_width=True, key="clear_logs"):
            # Limpar a fila de status
            while not scheduler.status_queue.empty():
                scheduler.status_queue.get_nowait()
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_header():
    """Exibe o cabeçalho da aplicação"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown('<h1 style="margin-bottom: 0.5rem;">⚽ Previsões SofaScore</h1>', unsafe_allow_html=True)
        st.markdown('<p style="margin: 0;">Sistema Inteligente de Previsão de Futebol - Atualização Automática</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.write("")
        st.write("")
        atualizar = st.button("🔄 Atualizar Dados", use_container_width=True, help="Atualiza apenas a visualização")
        if atualizar:
            st.rerun()
    
    with col3:
        st.write("")
        st.write("")
        agora = datetime.now().strftime('%H:%M')
        st.metric("Hora Local", agora)

def display_sidebar(data_manager, admin_mode=False):
    """Cria a sidebar com filtros e informações"""
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        # Modo Admin
        if admin_mode:
            display_scheduler_status(admin_mode=True)
        else:
            # Para usuários normais, mostrar apenas status básico
            if 'scheduler' in st.session_state:
                scheduler = st.session_state.scheduler
                if scheduler.last_run:
                    st.caption(f"📅 Última atualização: {scheduler.last_run.strftime('%H:%M:%S')}")
        
        # Informações do sistema
        st.markdown("### 📊 Estatísticas Rápidas")
        if data_manager.data is not None and not data_manager.data.empty:
            total_jogos = len(data_manager.data)
            
            if 'Tipo_Status_depois_previsao' in data_manager.data.columns:
                live_jogos = len(data_manager.data[
                    data_manager.data['Tipo_Status_depois_previsao'] == 'inprogress'
                ])
            else:
                live_jogos = 0
            
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
        
        # Filtro por status pós-previsão
        if 'Tipo_Status_depois_previsao' in data_manager.data.columns:
            status_opts = ['Todos'] + sorted(data_manager.data['Tipo_Status_depois_previsao'].dropna().unique().tolist())
            filters['tipo_status'] = st.selectbox("📊 Status", status_opts)
        
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
        
        # Botão para limpar filtros
        if st.button("🧹 Limpar Filtros", use_container_width=True):
            st.rerun()
        
        # Informações
        st.markdown("### ℹ️ Legenda")
        
        with st.expander("Ver estados"):
            st.markdown("""
            - **🟢 GREEN**: Previsão correta
            - **🔴 RED**: Previsão incorreta  
            - **⚪ PENDENTE**: Aguardando resultado
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

def display_compact_game_card(row, data_manager, show_chart=True):
    """Exibe um card compacto para o jogo"""
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Linha 1: Cabeçalho compacto
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        
        with col1:
            home = row.get('Time_Home', 'N/A')
            away = row.get('Time_Away', 'N/A')
            torneio = row.get('Torneio', '')
            
            st.markdown(f"**{home}** vs **{away}**")
            st.caption(f"🎯 {torneio} | {row.get('Data_Hora', '')}")
        
        with col2:
            estado = row.get('Estado_Previsao_Geral', 'pendente')
            if estado == 'green':
                st.markdown('<span class="estado-green">✅</span>', unsafe_allow_html=True)
            elif estado == 'red':
                st.markdown('<span class="estado-red">❌</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="estado-pendente">⏳</span>', unsafe_allow_html=True)
        
        with col3:
            confianca = row.get('nivel_confianca_ajustado', '')
            if 'Alta' in confianca:
                st.markdown('<span class="confidence-high">🟢</span>', unsafe_allow_html=True)
            elif 'Média' in confianca:
                st.markdown('<span class="confidence-medium">🟡</span>', unsafe_allow_html=True)
            elif 'Baixa' in confianca:
                st.markdown('<span class="confidence-low">🔴</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="confidence-very-low">⚪</span>', unsafe_allow_html=True)
        
        with col4:
            consenso = row.get('previsao_consensual_ajustada', 'Não')
            if consenso == 'Sim':
                st.markdown("✅")
            else:
                st.markdown("❌")
        
        # Linha 2: Dados do jogo compactos
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            placar_ht = row.get('PLACAR_HT', '0-0')
            st.markdown(f"**HT**")
            st.markdown(f"`{placar_ht}`")
        
        with col2:
            placar_ft = row.get('PLACAR_FT', '0-0')
            placar_depois = row.get('Placar_depois_previsao', placar_ft)
            st.markdown(f"**FT**")
            st.markdown(f"`{placar_depois}`")
        
        with col3:
            tipo_status = row.get('Tipo_Status_depois_previsao', '')
            minutos = row.get('Minutos_jogo_depois_previsao', row.get('Minutos_jogo', ''))
            
            st.markdown(f"**Status**")
            if tipo_status == 'inprogress':
                st.markdown(f"`LIVE`")
                if minutos:
                    st.caption(f"{minutos}'")
            elif tipo_status in ['ended', 'finished']:
                st.markdown(f"`TERMINADO`")
            else:
                st.markdown(f"`{tipo_status.upper()}`")
        
        with col4:
            # Links Bet365 compactos
            time_home = row.get('Time_Home', '')
            if time_home not in ['', 'N/A', '0']:
                time_home_clean = clean_team_name_for_url(time_home, to_lower=False)
                time_home_encoded = urllib.parse.quote(time_home_clean)
                
                st.markdown(f"**Links**")
                st.markdown(f'<a href="https://www.bet365.com/#/AX/K%5E{time_home_encoded}" target="_blank" class="bet365-link" style="font-size: 0.7rem; padding: 3px 6px;">Bet365</a>', unsafe_allow_html=True)
        
        # Detalhes expandíveis
        with st.expander("📊 Detalhes do Jogo", expanded=False):
            # Gráfico de Momentum (se disponível)
            if show_chart and data_manager.df_pontos is not None and not data_manager.df_pontos.empty:
                game_id = str(row.get('ID_Jogo', ''))
                momentum_data = get_momentum_data(game_id, data_manager.df_pontos)
                evolucao_text = row.get('evolução do Placar_depois_previsao', '')
                eventos_golos, placar_final = parse_evolucao_placar(evolucao_text)
                
                if momentum_data:
                    minutos_atual = row.get('Minutos_jogo_depois_previsao', '')
                    tipo_status = row.get('Tipo_Status_depois_previsao', '')
                    previsao_text = row.get("Previsao_Sim_Concatenado", "")
                    
                    st.markdown('<div class="matplotlib-container">', unsafe_allow_html=True)
                    
                    chart_html = create_matplotlib_momentum_chart(
                        game_id=game_id,
                        time_home=home,
                        time_away=away,
                        momentum_data=momentum_data,
                        eventos_golos=eventos_golos,
                        placar_final=placar_final,
                        status_jogo=tipo_status,
                        minutos_atual=minutos_atual,
                        previsao_text=previsao_text
                    )
                    
                    if chart_html:
                        st.markdown(chart_html, unsafe_allow_html=True)
                    else:
                        st.info("Gráfico não disponível")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Detalhes da previsão
            col1, col2 = st.columns(2)
            
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
                st.markdown(f"Concordância: `{concordancia}/5`")
                st.markdown(f"Score: `{score}`")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_live_games(data, data_manager):
    """Exibe jogos em andamento (Live)"""
    if data.empty:
        st.info("📭 Nenhum jogo live encontrado")
        return
    
    if 'Tipo_Status_depois_previsao' in data.columns:
        live_mask = data['Tipo_Status_depois_previsao'] == 'inprogress'
        live_data = data[live_mask].copy()
    else:
        live_data = pd.DataFrame()
    
    if live_data.empty:
        st.info("📭 Nenhum jogo em andamento no momento")
        return
    
    if 'Minutos_jogo_depois_previsao' in live_data.columns:
        def extract_minutes(x):
            try:
                if pd.isna(x):
                    return 0
                if str(x).lower() == 'terminado':
                    return 999
                parts = str(x).split(':')[0]
                return int(parts) if parts.isdigit() else 0
            except:
                return 0
        
        live_data['_minutos_num'] = live_data['Minutos_jogo_depois_previsao'].apply(extract_minutes)
        live_data = live_data.sort_values('_minutos_num', ascending=False)
    
    st.markdown(f"### 🔥 Jogos em Andamento ({len(live_data)})")
    
    show_charts = st.session_state.get('show_charts', True)
    
    for idx, row in live_data.iterrows():
        display_compact_game_card(row, data_manager, show_chart=show_charts)

def display_finished_games(data, data_manager):
    """Exibe jogos terminados"""
    if data.empty:
        st.info("📭 Nenhum jogo terminado encontrado")
        return
    
    if 'Tipo_Status_depois_previsao' in data.columns:
        finished_mask = data['Tipo_Status_depois_previsao'].isin(['ended', 'finished', 'ft'])
        finished_data = data[finished_mask].copy()
    else:
        finished_data = pd.DataFrame()
    
    if finished_data.empty:
        st.info("📭 Nenhum jogo terminado encontrado")
        return
    
    if 'Timestamp' in finished_data.columns:
        finished_data = finished_data.sort_values('Timestamp', ascending=False)
    
    st.markdown(f"### ✅ Jogos Terminados ({len(finished_data)})")
    
    show_charts = st.session_state.get('show_charts', True)
    
    # Agrupar por status se disponível
    if 'Tipo_Status_depois_previsao' in finished_data.columns:
        status_groups = finished_data['Tipo_Status_depois_previsao'].unique()
        
        for status in status_groups:
            if pd.isna(status) or status == '':
                continue
                
            group_data = finished_data[finished_data['Tipo_Status_depois_previsao'] == status]
            
            # Não usar expander dentro de expander - mostrar diretamente
            st.markdown(f"#### {status.upper()} ({len(group_data)} jogos)")
            
            # Mostrar cards dos jogos
            for idx, row in group_data.iterrows():
                display_compact_game_card(row, data_manager, show_chart=show_charts)
    else:
        # Sem agrupamento por status
        for idx, row in finished_data.iterrows():
            display_compact_game_card(row, data_manager, show_chart=show_charts)

def display_analytics_tab(data, data_manager):
    """Exibe a aba de análises"""
    if data.empty:
        st.info("📭 Dados insuficientes para análise")
        return
    
    st.markdown("## 📈 Análises e Estatísticas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Estatísticas", "🎯 Performance", "📅 Temporal", "🔍 Detalhes"])
    
    with tab1:
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance por Confiança")
            if 'nivel_confianca_ajustado' in data.columns and 'Estado_Previsao_Geral' in data.columns:
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
        st.markdown("#### 📋 Dados Detalhados")
        
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
        
        #### 🔄 Atualização Automática
        
        O sistema inclui um scheduler que executa automaticamente todos os scripts periodicamente.
        A atualização é automática e não requer intervenção do usuário.
        
        #### 📊 Gráficos de Momentum
        
        Os gráficos de momentum mostram a evolução do domínio do jogo ao longo dos minutos:
        - **🔵 Azul**: Gol do time da casa
        - **🟠 Laranja**: Gol do time visitante
        - **📊 Barras verdes**: Momentum positivo
        - **📊 Barras vermelhas**: Momentum negativo
        """)
    
    with col2:
        st.markdown("""
        #### 🏆 Estatísticas Atuais
        """)
        
        st.metric("Versão", "4.3.0")
        st.metric("Última Atualização", datetime.now().strftime('%d/%m/%Y'))
        st.metric("Modo", "Público")
        
        st.markdown("""
        #### 📞 Suporte
        
        **Problemas Comuns:**
        1. Dados não carregando
        2. Atualizações atrasadas
        3. Previsões inconsistentes
        
        **Soluções:**
        1. Aguardar próxima atualização automática
        2. Verificar conexão com internet
        3. Relatar ao administrador
        
        #### 🔄 Atualização
        
        O sistema é atualizado automaticamente.
        Para suporte técnico, contate o administrador.
        """)

def main():
    """Função principal da aplicação"""
    
    # Inicializar session state
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    
    # Inicializar scheduler (sem interface para usuário normal)
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = SchedulerService(interval_minutes=1)
        st.session_state.scheduler.start()
    
    # Verificar modo admin (senha do arquivo config)
    admin_mode = False
    with st.sidebar:
        if st.checkbox("🔐 Modo Administrador"):
            password = st.text_input("Senha", type="password")
            if password == ADMIN_PASSWORD:
                admin_mode = True
                st.success("Modo administrador ativado")
            elif password:
                st.error("Senha incorreta")
    
    # Inicializar gerenciador de dados
    data_manager = DataManager()
    
    # Header principal
    display_header()
    
    if data_manager.data is None or data_manager.data.empty:
        st.warning("⚠️ Não foi possível carregar os dados. Aguarde a próxima atualização automática.")
        
        if admin_mode:
            with st.expander("📝 Informações Técnicas (Admin)"):
                st.markdown("""
                1. **Verificar scripts:**
                ```bash
                ls /workspaces/previsao_sofascore/scripts/
                ```
                
                2. **Verificar dados:**
                ```bash
                ls /workspaces/previsao_sofascore/data/
                ```
                
                3. **Logs do scheduler:**
                Verifique os logs na seção de administrador.
                """)
        return
    
    # Sidebar com filtros
    with st.sidebar:
        filters = display_sidebar(data_manager, admin_mode=admin_mode)
    
    # Aplicar filtros
    filtered_data = apply_filters(data_manager.data, filters)
    
    # Métricas horizontais modernas
    display_horizontal_metrics(filtered_data)
    
    # Controle de gráficos
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 📊 Jogos Filtrados: {len(filtered_data)}")
    with col2:
        show_charts = st.checkbox("📈 Mostrar Gráficos", value=st.session_state.get('show_charts', True))
        if show_charts != st.session_state.get('show_charts'):
            st.session_state.show_charts = show_charts
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Live", "✅ Terminados", "📈 Análises", "ℹ️ Sobre"])
    
    with tab1:
        display_live_games(filtered_data, data_manager)
    
    with tab2:
        display_finished_games(filtered_data, data_manager)
    
    with tab3:
        display_analytics_tab(filtered_data, data_manager)
    
    with tab4:
        display_about_tab()

# Ponto de entrada principal
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()