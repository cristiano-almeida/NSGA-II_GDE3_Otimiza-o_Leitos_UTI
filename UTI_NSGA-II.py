import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Any

# Importações do Pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ---------------------------------------------------------------------------

def configurar_logging(nivel_logging: str = 'INFO') -> logging.Logger:
    """Configura e retorna logger para monitoramento detalhado."""
    logger = logging.getLogger('OtimizacaoUTI')
    logger.setLevel(getattr(logging, nivel_logging))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# ---------------------------------------------------------------------------
# 1. GERAÇÃO E VALIDAÇÃO DE DADOS
# ---------------------------------------------------------------------------

def gerar_dados_simulacao_realista(numero_pacientes: int, horizonte_tempo: int, logger: logging.Logger) -> List[Dict]:
    """
    Gera dados de pacientes realistas para simulação de UTI com distribuições
    baseadas em padrões clínicos reais.
    """
    np.random.seed(42)
    
    pacientes = []
    for i in range(numero_pacientes):
        # Tempo de chegada: distribuído exponencialmente (mais chegadas no início)
        tempo_chegada = int(np.random.exponential(scale=24))
        tempo_chegada = min(tempo_chegada, horizonte_tempo // 2)
        
        # Gravidade: distribuição realista (mais pacientes moderados)
        gravidade = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                   p=[0.05, 0.10, 0.15, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])
        
        # Tempo estimado UTI baseado na gravidade (mais realista)
        if gravidade >= 8:  # Pacientes críticos
            tempo_uti = np.random.randint(72, 120)  # 3-5 dias
        elif gravidade >= 5:  # Pacientes moderados
            tempo_uti = np.random.randint(48, 96)   # 2-4 dias
        else:  # Pacientes leves
            tempo_uti = np.random.randint(24, 72)   # 1-3 dias
        
        pacientes.append({
            "id_paciente": i + 1,
            "tempo_chegada_hora": tempo_chegada,
            "gravidade_score": gravidade,
            "tempo_estimado_uti_horas": tempo_uti
        })
    
    logger.info(f"Dados gerados: {len(pacientes)} pacientes com distribuição realista")
    return pacientes

def carregar_ou_gerar_dados(nome_arquivo: str, numero_pacientes: int, horizonte_tempo: int, logger: logging.Logger) -> List[Dict]:
    """Carrega dados existentes ou gera nova base."""
    if os.path.exists(nome_arquivo):
        logger.info(f"Carregando base de dados existente: '{nome_arquivo}'")
        df_pacientes = pd.read_csv(nome_arquivo)
        pacientes = df_pacientes.to_dict('records')
    else:
        logger.info("Gerando nova base de dados realista...")
        pacientes = gerar_dados_simulacao_realista(numero_pacientes, horizonte_tempo, logger)
        df_pacientes = pd.DataFrame(pacientes)
        df_pacientes.to_csv(nome_arquivo, index=False, encoding='utf-8')
        logger.info(f"Base de dados salva em: '{nome_arquivo}'")
    
    return pacientes

def ajustar_tempos_uti_para_limites_realistas(pacientes: List[Dict], limite_maximo_horas: int = 120, logger: logging.Logger = None) -> List[Dict]:
    """Ajusta tempos de UTI para limites clinicamente realistas."""
    pacientes_ajustados = []
    alteracoes_realizadas = 0
    
    for paciente in pacientes:
        tempo_original = paciente['tempo_estimado_uti_horas']
        tempo_ajustado = min(tempo_original, limite_maximo_horas)
        
        if tempo_original != tempo_ajustado:
            alteracoes_realizadas += 1
            if logger:
                logger.info(f"Paciente {paciente['id_paciente']}: Tempo UTI ajustado de {tempo_original} para {tempo_ajustado} horas")
        
        pacientes_ajustados.append({
            'id_paciente': paciente['id_paciente'],
            'tempo_chegada_hora': paciente['tempo_chegada_hora'],
            'gravidade_score': paciente['gravidade_score'],
            'tempo_estimado_uti_horas': tempo_ajustado
        })
    
    if logger and alteracoes_realizadas > 0:
        logger.info(f"Tempos de UTI ajustados: {alteracoes_realizadas} pacientes modificados")
        logger.info(f"Limite máximo estabelecido: {limite_maximo_horas} horas ({limite_maximo_horas//24} dias)")
    
    return pacientes_ajustados

def verificar_viabilidade_sistema(pacientes: List[Dict], numero_leitos: int, horizonte_tempo: int, logger: logging.Logger) -> Dict[str, Any]:
    """Verifica a viabilidade do sistema e fornece estatísticas detalhadas."""
    capacidade_total = numero_leitos * horizonte_tempo
    tempo_total_requerido = sum(paciente['tempo_estimado_uti_horas'] for paciente in pacientes)
    taxa_ocupacao_teorica = (tempo_total_requerido / capacidade_total) * 100 if capacidade_total > 0 else 0
    
    analise = {
        'numero_leitos': numero_leitos,
        'horizonte_tempo': horizonte_tempo,
        'capacidade_total': capacidade_total,
        'tempo_total_requerido': tempo_total_requerido,
        'taxa_ocupacao_teorica': taxa_ocupacao_teorica,
        'numero_pacientes': len(pacientes),
        'viavel': taxa_ocupacao_teorica <= 95
    }
    
    logger.info("=" * 80)
    logger.info("ANÁLISE DE VIABILIDADE DO CENÁRIO")
    logger.info("=" * 80)
    logger.info(f"Número de leitos: {numero_leitos}")
    logger.info(f"Horizonte de planejamento: {horizonte_tempo} horas ({horizonte_tempo//24} dias)")
    logger.info(f"Capacidade total do sistema: {capacidade_total} horas-leito")
    logger.info(f"Demanda total dos pacientes: {tempo_total_requerido} horas-leito")
    logger.info(f"Taxa de ocupação teórica: {taxa_ocupacao_teorica:.1f}%")
    
    if taxa_ocupacao_teorica > 95:
        logger.warning("AVISO: Sistema com alta probabilidade de ser super-restringido.")
        logger.warning("Considere aumentar o número de leitos ou o horizonte temporal.")
    elif taxa_ocupacao_teorica > 85:
        logger.info("Sistema com boa capacidade mas próximo do limite.")
    else:
        logger.info("Sistema com capacidade adequada.")
    
    return analise

def calcular_valores_referencia_realistas(pacientes: List[Dict], horizonte_tempo: int, logger: logging.Logger) -> Tuple[float, float, float]:
    """Calcula valores de referência realistas para normalização."""
    tempo_espera_maximo = 0
    risco_clinico_maximo = 0
    custo_terminal_maximo = 0
    
    for paciente in pacientes:
        tempo_espera_max_paciente = min(horizonte_tempo - paciente['tempo_chegada_hora'], 
                                      paciente['tempo_estimado_uti_horas'])
        tempo_espera_maximo += tempo_espera_max_paciente
        risco_clinico_maximo += tempo_espera_max_paciente * paciente['gravidade_score']
        
        if paciente['tempo_chegada_hora'] + paciente['tempo_estimado_uti_horas'] > horizonte_tempo:
            excesso_maximo = (paciente['tempo_chegada_hora'] + paciente['tempo_estimado_uti_horas'] - horizonte_tempo)
            custo_terminal_maximo += excesso_maximo * paciente['gravidade_score']
    
    tempo_espera_maximo = max(tempo_espera_maximo, 1)
    risco_clinico_maximo = max(risco_clinico_maximo, 1)
    custo_terminal_maximo = max(custo_terminal_maximo, 1)
    
    logger.info("VALORES DE REFERÊNCIA REALISTAS:")
    logger.info(f"Tempo de espera máximo de referência: {tempo_espera_maximo:.0f} horas")
    logger.info(f"Risco clínico máximo de referência: {risco_clinico_maximo:.0f}")
    logger.info(f"Custo terminal máximo de referência: {custo_terminal_maximo:.0f}")
    
    return tempo_espera_maximo, risco_clinico_maximo, custo_terminal_maximo

def calcular_ocupacao_precisa(tempo_inicio_uti: int, tempo_fim_uti: int, horizonte_tempo: int) -> np.ndarray:
    """Calcula a ocupação de leitos com precisão."""
    ocupacao = np.zeros(horizonte_tempo, dtype=int)
    inicio = max(0, tempo_inicio_uti)
    fim = min(horizonte_tempo, tempo_fim_uti)
    
    if fim > inicio:
        ocupacao[inicio:fim] += 1
    
    return ocupacao

# ---------------------------------------------------------------------------
# 2. DEFINIÇÃO DO PROBLEMA DE OTIMIZAÇÃO (COMBINAÇÃO DAS MELHORES ABORDAGENS)
# ---------------------------------------------------------------------------

class ProblemaOtimizacaoUTI(ElementwiseProblem):
    def __init__(self, pacientes: List[Dict], numero_leitos: int, horizonte_tempo: int, logger: logging.Logger):
        self.pacientes = pacientes
        self.numero_leitos = numero_leitos
        self.horizonte_tempo = horizonte_tempo
        self.numero_pacientes = len(pacientes)
        self.logger = logger
        
        self.tempo_espera_max_ref, self.risco_clinico_max_ref, self.custo_terminal_max_ref = \
            calcular_valores_referencia_realistas(pacientes, horizonte_tempo, logger)
        
        limites_inferiores = [paciente['tempo_chegada_hora'] for paciente in pacientes]
        limites_superiores = [horizonte_tempo for _ in pacientes]
        
        super().__init__(
            n_var=self.numero_pacientes, 
            n_obj=4,
            n_constr=2,  # Restrições rígidas (capacidade e precedência)
            xl=np.array(limites_inferiores), 
            xu=np.array(limites_superiores),
            vtype=int 
        )
        
        self.logger.info(f"Problema de otimização inicializado com {self.numero_pacientes} variáveis")

    def _evaluate(self, x, out, *args, **kwargs):
        cronograma = np.round(x).astype(int)
        
        tempo_espera_total = 0
        risco_clinico_total = 0
        custo_ocupacao_terminal = 0
        uso_leitos_por_hora = np.zeros(self.horizonte_tempo, dtype=int)
        
        for indice, paciente in enumerate(self.pacientes):
            tempo_inicio_uti = cronograma[indice]
            
            tempo_espera = max(0, tempo_inicio_uti - paciente['tempo_chegada_hora'])
            tempo_espera_total += tempo_espera
            risco_clinico_total += tempo_espera * paciente['gravidade_score']
            
            tempo_fim_uti = tempo_inicio_uti + paciente['tempo_estimado_uti_horas']
            
            if tempo_fim_uti > self.horizonte_tempo:
                excesso_tempo = tempo_fim_uti - self.horizonte_tempo
                custo_ocupacao_terminal += excesso_tempo * paciente['gravidade_score']
            
            uso_leitos_por_hora += calcular_ocupacao_precisa(tempo_inicio_uti, tempo_fim_uti, self.horizonte_tempo)
        
        # Restrições rígidas
        maxima_ocupacao = np.max(uso_leitos_por_hora) if self.horizonte_tempo > 0 else 0
        violacao_capacidade = max(0, maxima_ocupacao - self.numero_leitos)
        
        violacao_precedencia = sum(max(0, p['tempo_chegada_hora'] - cronograma[i]) for i, p in enumerate(self.pacientes))

        out["G"] = [violacao_capacidade, violacao_precedencia]
        
        # Objetivos normalizados com valores de referência realistas
        objetivo_tempo_espera = tempo_espera_total / self.tempo_espera_max_ref
        
        utilizacao_media = np.mean(uso_leitos_por_hora)
        taxa_utilizacao = utilizacao_media / self.numero_leitos if self.numero_leitos > 0 else 0
        objetivo_utilizacao = 1 - taxa_utilizacao
        
        objetivo_risco_clinico = risco_clinico_total / self.risco_clinico_max_ref
        objetivo_custo_terminal = custo_ocupacao_terminal / self.custo_terminal_max_ref
        
        out["F"] = [objetivo_tempo_espera, objetivo_utilizacao, objetivo_risco_clinico, objetivo_custo_terminal]

# ---------------------------------------------------------------------------
# 3. FUNÇÕES PARA ANÁLISE E RELATÓRIOS
# ---------------------------------------------------------------------------

def gerar_relatorio_detalhado_plano(pacientes: List[Dict], cronograma: np.ndarray, nome_arquivo_saida: str, 
                                  numero_leitos: int, horizonte_tempo: int, logger: logging.Logger) -> Dict[str, Any]:
    """Gera relatório completo do plano de alocação."""
    cronograma_inteiro = np.round(cronograma).astype(int)

    dados_relatorio = []
    metricas = {
        'tempo_espera_total': 0, 'risco_clinico_total': 0, 'custo_terminal_total': 0,
        'pacientes_fora_horizonte': 0, 'pacientes_com_espera': 0, 'pacientes_imediatos': 0,
        'utilizacao_media_leitos': 0, 'maxima_ocupacao_simultanea': 0, 'violacao_capacidade': 0,
        'horas_ociosas_total': 0
    }
    
    ocupacao_horaria = np.zeros(horizonte_tempo, dtype=int)
    
    for indice, paciente in enumerate(pacientes):
        hora_internacao = cronograma_inteiro[indice]
        tempo_espera = max(0, hora_internacao - paciente['tempo_chegada_hora'])
        hora_alta = hora_internacao + paciente['tempo_estimado_uti_horas']
        custo_terminal = max(0, hora_alta - horizonte_tempo) * paciente['gravidade_score']
        
        if hora_alta > horizonte_tempo:
            metricas['pacientes_fora_horizonte'] += 1
        
        if tempo_espera > 0:
            metricas['pacientes_com_espera'] += 1
        else:
            metricas['pacientes_imediatos'] += 1

        ocupacao_horaria += calcular_ocupacao_precisa(hora_internacao, hora_alta, horizonte_tempo)
        
        dados_relatorio.append({
            "ID_Paciente": paciente['id_paciente'],
            "Hora_Chegada": paciente['tempo_chegada_hora'],
            "Gravidade_Score": paciente['gravidade_score'],
            "Tempo_UTI_Estimado": paciente['tempo_estimado_uti_horas'],
            "Hora_Internacao_Otimizada": hora_internacao,
            "Tempo_Espera_Horas": tempo_espera,
            "Hora_Estimada_Alta": hora_alta,
            "Custo_Terminal": custo_terminal,
            "Status": "Fora do Horizonte" if hora_alta > horizonte_tempo else "Dentro do Horizonte",
            "Internacao_Imediata": "Sim" if tempo_espera == 0 else "Não"
        })
        
        metricas['tempo_espera_total'] += tempo_espera
        metricas['risco_clinico_total'] += tempo_espera * paciente['gravidade_score']
        metricas['custo_terminal_total'] += custo_terminal

    metricas['utilizacao_media_leitos'] = np.mean(ocupacao_horaria)
    metricas['maxima_ocupacao_simultanea'] = np.max(ocupacao_horaria)
    metricas['violacao_capacidade'] = max(0, metricas['maxima_ocupacao_simultanea'] - numero_leitos)
    metricas['horas_ociosas_total'] = sum(max(0, numero_leitos - ocupacao) for ocupacao in ocupacao_horaria)
    
    df_relatorio = pd.DataFrame(dados_relatorio)
    df_relatorio.to_csv(nome_arquivo_saida, index=False, encoding='utf-8')
    
    logger.info(f"Relatório detalhado salvo em: '{nome_arquivo_saida}'")
    logger.info(f"  - Tempo de Espera Total: {metricas['tempo_espera_total']:.0f} horas")
    logger.info(f"  - Utilização Média: {metricas['utilizacao_media_leitos']:.2f} leitos ({(metricas['utilizacao_media_leitos']/numero_leitos*100):.1f}%)")
    logger.info(f"  - Ocupação Máxima: {metricas['maxima_ocupacao_simultanea']} leitos")
    logger.info(f"  - Violação de Capacidade: {metricas['violacao_capacidade']} leitos")
    logger.info(f"  - Pacientes Fora do Horizonte: {metricas['pacientes_fora_horizonte']}")
    
    return metricas

def analisar_resultados_otimizacao(resultado, problema, pacientes, numero_leitos, horizonte_tempo, logger):
    """Analisa os resultados da otimização detalhadamente."""
    if resultado.F is not None and len(resultado.F) > 0:
        num_solucoes = len(resultado.F)
        logger.info(f"Número de soluções viáveis encontradas na fronteira de Pareto: {num_solucoes}")
        
        solucoes_unicas = len(np.unique(resultado.X, axis=0))
        logger.info(f"Soluções únicas: {solucoes_unicas}")
        
        fronteira_pareto = resultado.F
        tempo_espera_real = fronteira_pareto[:, 0] * problema.tempo_espera_max_ref
        utilizacao_real = 1 - fronteira_pareto[:, 1]
        risco_clinico_real = fronteira_pareto[:, 2] * problema.risco_clinico_max_ref
        custo_terminal_real = fronteira_pareto[:, 3] * problema.custo_terminal_max_ref

        min_espera, max_espera = np.min(tempo_espera_real), np.max(tempo_espera_real)
        
        logger.info("Variação entre as soluções na fronteira de Pareto:")
        logger.info(f"Tempo de espera: De {min_espera:.0f} a {max_espera:.0f} horas")
        logger.info(f"Utilização: De {np.min(utilizacao_real)*100:.1f}% a {np.max(utilizacao_real)*100:.1f}%")
        logger.info(f"Risco clínico: De {np.min(risco_clinico_real):.0f} a {np.max(risco_clinico_real):.0f}")
        logger.info(f"Custo terminal: De {np.min(custo_terminal_real):.0f} a {np.max(custo_terminal_real):.0f}")
        
        if (max_espera - min_espera) < 10:
            logger.warning("AVISO: A fronteira de Pareto colapsou. O problema parece ser super-restringido.")
        else:
            logger.info("Boa diversidade de soluções encontrada na fronteira de Pareto.")

        indices_otimos = {
            'menor_tempo_espera': np.argmin(tempo_espera_real),
            'maior_utilizacao': np.argmax(utilizacao_real),
            'menor_risco_clinico': np.argmin(risco_clinico_real),
            'menor_custo_terminal': np.argmin(custo_terminal_real),
            'solucao_balanceada': np.argmin(np.sum(fronteira_pareto, axis=1))
        }
        
        logger.info("SOLUÇÕES DE DESTAQUE ENCONTRADAS (TRADE-OFFS):")
        logger.info("-" * 100)
        
        for criterio, indice in indices_otimos.items():
            logger.info(f"{criterio.upper().replace('_', ' '):<25}: "
                      f"Espera={tempo_espera_real[indice]:.0f}h, "
                      f"Utilização={(utilizacao_real[indice]*100):.1f}%, "
                      f"Risco={risco_clinico_real[indice]:.0f}, "
                      f"Custo={custo_terminal_real[indice]:.0f}")
        
        logger.info("GERANDO RELATÓRIOS DETALHADOS...")
        for criterio, indice in indices_otimos.items():
            nome_arquivo = f"relatorio_{criterio}.csv"
            logger.info(f"Gerando relatório para '{criterio}'...")
            metricas = gerar_relatorio_detalhado_plano(pacientes, resultado.X[indice], nome_arquivo, numero_leitos, horizonte_tempo, logger)
        
        return True
    
    else:
        logger.error("NENHUMA SOLUÇÃO VIÁVEL FOI ENCONTRADA.")
        logger.error("Causas prováveis:")
        logger.error("  1. O problema é super-restringido")
        logger.error("  2. Parâmetros do algoritmo insuficientes")
        logger.error("Sugestões: Aumentar leitos, horizonte ou parâmetros do algoritmo")
        return False

def visualizar_fronteira_pareto(resultado, problema, logger):
    """Gera visualização mais intuitiva da fronteira de Pareto."""
    try:
        if resultado.F is not None and len(resultado.F) > 0:
            # Prepara dados - CORRIGINDO a direção dos objetivos
            objetivos = {
                'Tempo Espera (h)': resultado.F[:, 0] * problema.tempo_espera_max_ref,
                'Utilização Leitos (%)': (1 - resultado.F[:, 1]) * 100,  # Agora: maior = melhor
                'Risco Clínico': resultado.F[:, 2] * problema.risco_clinico_max_ref,
                'Custo Terminal': resultado.F[:, 3] * problema.custo_terminal_max_ref
            }
            
            # Normaliza para escala 0-1 (1 = SEMPRE melhor)
            objetivos_normalizados = {}
            for nome, valores in objetivos.items():
                if 'Utilização' in nome:
                    # MAXIMIZAÇÃO: 1 = melhor (100% utilização)
                    objetivos_normalizados[nome] = valores / 100
                else:
                    # MINIMIZAÇÃO: inverte para 1 = melhor
                    max_val = np.max(valores)
                    min_val = np.min(valores)
                    objetivos_normalizados[nome] = 1 - (valores - min_val) / (max_val - min_val)
            
            # Identifica soluções de destaque nos valores ORIGINAIS
            indices_otimos = {
                'Menor Tempo Espera': np.argmin(objetivos['Tempo Espera (h)']),
                'Maior Utilização': np.argmax(objetivos['Utilização Leitos (%)']),
                'Menor Risco': np.argmin(objetivos['Risco Clínico']),
                'Menor Custo Terminal': np.argmin(objetivos['Custo Terminal']),
                'Solução Balanceada': np.argmin(np.sum(resultado.F, axis=1))
            }
            
            # Configuração do gráfico
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # GRÁFICO 1: Valores normalizados (1 = sempre melhor)
            nomes_objetivos = list(objetivos_normalizados.keys())
            x_pos = range(len(nomes_objetivos))
            
            # Todas as soluções (cinza)
            for i in range(len(resultado.F)):
                valores = [objetivos_normalizados[nome][i] for nome in nomes_objetivos]
                ax1.plot(x_pos, valores, color='lightgray', linewidth=0.5, alpha=0.6)
            
            # Soluções de destaque (coloridas)
            cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            for j, (label, idx) in enumerate(indices_otimos.items()):
                valores = [objetivos_normalizados[nome][idx] for nome in nomes_objetivos]
                ax1.plot(x_pos, valores, marker='o', linewidth=2.5, 
                        color=cores[j], label=label, markersize=8)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(nomes_objetivos, rotation=45)
            ax1.set_ylabel('Desempenho Normalizado\n(1 = Melhor, 0 = Pior)')
            ax1.set_title('Fronteira de Pareto - Visão Normalizada')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_ylim(0, 1)
            
            # GRÁFICO 2: Valores reais (para contexto)
            for j, (label, idx) in enumerate(indices_otimos.items()):
                valores_reais = [objetivos[nome][idx] for nome in nomes_objetivos]
                ax2.plot(x_pos, valores_reais, marker='s', linewidth=2, 
                        color=cores[j], label=label, markersize=6)
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(nomes_objetivos, rotation=45)
            ax2.set_ylabel('Valores Reais')
            ax2.set_title('Valores Absolutos das Soluções de Destaque')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Ajustes finais
            plt.tight_layout()
            
            # Salvar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"fronteira_pareto_melhorada_{timestamp}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualização melhorada salva em '{nome_arquivo}'")
            
    except Exception as e:
        logger.error(f"Erro na geração do gráfico: {str(e)}")

# ---------------------------------------------------------------------------
# 4. SIMULAÇÃO PRINCIPAL COM PARÂMETROS OTIMIZADOS
# ---------------------------------------------------------------------------

def executar_simulacao_completa():
    """Executa a simulação completa com parâmetros otimizados."""
    # --- PARÂMETROS OTIMIZADOS ---
    NUMERO_LEITOS_UTI = 12
    HORIZONTE_PLANEJAMENTO_HORAS = 168
    NUMERO_PACIENTES = 30
    NOME_ARQUIVO_BASE = "base_pacientes_uti_realista.csv"
    
    # PARÂMETROS DO ALGORITMO OTIMIZADOS
    NUMERO_GERACOES = 250
    TAMANHO_POPULACAO = 350
    LIMITE_MAXIMO_UTI = 120  # 5 dias
    
    logger = configurar_logging('INFO')
    
    try:
        logger.info("=" * 100)
        logger.info("SISTEMA DE OTIMIZAÇÃO DE ALOCAÇÃO DE LEITOS DE UTI - VERSÃO OTIMIZADA")
        logger.info("=" * 100)
        
        # Carregar ou gerar dados
        pacientes = carregar_ou_gerar_dados(NOME_ARQUIVO_BASE, NUMERO_PACIENTES, HORIZONTE_PLANEJAMENTO_HORAS, logger)
        pacientes = ajustar_tempos_uti_para_limites_realistas(pacientes, LIMITE_MAXIMO_UTI, logger)
        analise_viabilidade = verificar_viabilidade_sistema(pacientes, NUMERO_LEITOS_UTI, HORIZONTE_PLANEJAMENTO_HORAS, logger)
        
        if not analise_viabilidade['viavel']:
            logger.warning("Sistema pode ser inviável. Considere ajustar parâmetros.")
        
        problema = ProblemaOtimizacaoUTI(pacientes, NUMERO_LEITOS_UTI, HORIZONTE_PLANEJAMENTO_HORAS, logger)
        
        # Algoritmo com parâmetros otimizados
        algoritmo = NSGA2(
            pop_size=TAMANHO_POPULACAO,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        termination = ('n_gen', NUMERO_GERACOES)
        
        logger.info("INICIANDO PROCESSO DE OTIMIZAÇÃO...")
        logger.info(f"Parâmetros: {TAMANHO_POPULACAO} indivíduos, {NUMERO_GERACOES} gerações")
        
        resultado = minimize(
            problema, 
            algoritmo, 
            termination,
            verbose=True,
            seed=42,
            save_history=True
        )
        
        logger.info("OTIMIZAÇÃO CONCLUÍDA. ANALISANDO RESULTADOS...")
        
        sucesso = analisar_resultados_otimizacao(resultado, problema, pacientes, NUMERO_LEITOS_UTI, HORIZONTE_PLANEJAMENTO_HORAS, logger)
        visualizar_fronteira_pareto(resultado, problema, logger)
        
        if sucesso:
            logger.info("SIMULAÇÃO FINALIZADA COM SUCESSO!")
            logger.info("ESTATÍSTICAS FINAIS:")
            logger.info(f"Número total de avaliações: {resultado.algorithm.evaluator.n_eval}")
            logger.info(f"Tempo de execução: {resultado.exec_time:.2f} segundos")
            logger.info(f"Melhor solução encontrada: {len(resultado.F)} alternativas Pareto-ótimas")
            
        return resultado, sucesso
        
    except Exception as e:
        logger.error(f"Erro durante a simulação: {str(e)}")
        return None, False

# ---------------------------------------------------------------------------
# 5. EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    resultado, sucesso = executar_simulacao_completa()
    
    if sucesso:
        print("\n" + "="*80)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print("Relatórios detalhados gerados para análise.")
    else:
        print("\n" + "="*80)
        print("PROCESSO CONCLUÍDO COM ERROS!")
        print("="*80)
        print("Consulte os logs para detalhes.")