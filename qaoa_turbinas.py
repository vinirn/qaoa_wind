#!/usr/bin/env python3
"""
QAOA para Otimização de Turbinas Eólicas - Qiskit 2.x
Implementação moderna usando as APIs mais recentes
Sistema configurável via arquivo JSON
"""

import numpy as np
import json
import os
import sys
import csv
from datetime import datetime
from utils import (parse_arguments, list_available_configs, load_config, get_config_file,
                   validate_constraints, evaluate_solution, show_active_penalties, 
                   bitstring_to_grid, display_grid, display_interference_matrix, plot_cost_evolution,
                   plot_gamma_beta_trajectory, create_grid_visualization,
                   load_ibm_api_key, load_ibm_config, confirm_ibm_execution, plot_quantum_circuit)

# Controle global simples para gating de plots quando função utilitária é usada
PLOT_ENABLED = False
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

class QAOATurbineOptimizer:
    def __init__(self, config_file="config.json"):
        """Inicializa o otimizador com configurações do arquivo"""
        self.config = load_config(config_file)
        
        # Extrair parâmetros principais
        self.rows = self.config["grid"]["rows"]
        self.cols = self.config["grid"]["cols"]
        self.n_positions = self.rows * self.cols
        self.wind_direction = tuple(self.config["wind"]["direction"])
        # Suporte para nomenclatura antiga e nova
        if "wake_effects" in self.config:
            self.max_penalty = self.config["wake_effects"]["base_penalty"]
            self.decay_factor = self.config["wake_effects"]["distance_decay"]
        else:
            # Compatibilidade com configs antigos
            self.max_penalty = self.config["penalties"]["max_penalty"]
            self.decay_factor = self.config["penalties"]["decay_factor"]
        
        # NOVO: Restrições de turbinas
        constraints = self.config.get("constraints", {})
        
        # Tratar valores null corretamente
        min_turbines_raw = constraints.get("min_turbines", None)
        max_turbines_raw = constraints.get("max_turbines", None)
        
        self.min_turbines = min_turbines_raw if min_turbines_raw is not None else 0
        self.max_turbines = max_turbines_raw if max_turbines_raw is not None else self.n_positions
        self.enforce_constraints = constraints.get("enforce_constraints", False)
        
        # Buscar constraint_penalty em constraints primeiro, depois em penalties
        self.constraint_penalty = constraints.get("constraint_penalty", 
                                                 self.config.get("penalties", {}).get("constraint_penalty", 10.0))
        
        print("=== QAOA Configurável para Turbinas Eólicas ===\n")
        print(f"📐 Grid: {self.rows}x{self.cols} ({self.n_positions} posições)")
        print(f"🌬️  Vento: {self.wind_direction}")
        print(f"⚡ Penalidade base para esteira: {self.max_penalty}")
        
        # NOVO: Mostrar restrições de forma destacada
        print(f"\n🎯 RESTRIÇÕES DE NÚMERO DE TURBINAS:")
        if self.enforce_constraints:
            print(f"   ✅ Restrições ATIVAS")
            print(f"   📊 Mínimo: {self.min_turbines} turbinas")
            print(f"   📊 Máximo: {self.max_turbines} turbinas")
            if self.min_turbines == self.max_turbines:
                print(f"   🎯 Número EXATO exigido: {self.min_turbines} turbinas")
            print(f"   ⚠️  Penalidade por violação: {self.constraint_penalty}")
        else:
            print(f"   ❌ Restrições DESATIVADAS")
            print(f"   📊 Mínimo: 0 turbinas (sem restrição)")
            print(f"   📊 Máximo: {self.n_positions} turbinas (sem restrição)")
        self.setup_grid()
        self.setup_score()
        self.wake_penalties = self.calculate_wake_penalties()
        
        
    def setup_grid(self):
        """Configura as coordenadas do grid"""
        self.positions_coords = {}
        for i in range(self.n_positions):
            row = i // self.cols
            col = i % self.cols
            self.positions_coords[i] = (row, col)
            
    def setup_score(self):
        """Configura o score por posição"""
        score_config = self.config["score"]
        mode = score_config["mode"]
        
        if mode == "fixed":
            self.score = score_config["values"][:self.n_positions]
            # Preencher com zeros se necessário
            while len(self.score) < self.n_positions:
                self.score.append(0.0)
        elif mode == "random":
            min_val, max_val = score_config["random_range"]
            np.random.seed(42)  # Seed fixo para reprodutibilidade
            self.score = np.random.uniform(min_val, max_val, self.n_positions).tolist()
        elif mode == "uniform":
            uniform_value = score_config.get("uniform_value", 4.0)
            self.score = [uniform_value] * self.n_positions
        else:
            raise ValueError(f"Modo de score inválido: {mode}")
            
        print(f"💯 Score por posição: {[f'{s:.1f}' for s in self.score]}")
        
        
    def calculate_wake_penalties(self):
        """Calcula penalidades de esteira tradicional - mesma linha apenas"""
        penalties = {}
        
        for i in range(self.n_positions):
            for j in range(self.n_positions):
                if i != j:
                    coord1 = self.positions_coords[i]
                    coord2 = self.positions_coords[j]
                    
                    # Calcular diferenças baseadas na direção do vento
                    if self.wind_direction == (0, 1):  # oeste→leste
                        dx = coord2[1] - coord1[1]  # diferença leste-oeste
                        dy = coord2[0] - coord1[0]  # diferença norte-sul
                        
                        # Turbina j deve estar a leste de turbina i NA MESMA LINHA
                        if dx > 0 and dy == 0:  # mesma linha (dy = 0)
                            distance = abs(dx)
                            penalty = max(0, self.max_penalty - distance * self.decay_factor)
                            if penalty > 0:
                                penalties[(i, j)] = penalty
                    
                    elif self.wind_direction == (1, 0):  # norte→sul
                        dx = coord2[0] - coord1[0]  # diferença norte-sul
                        dy = coord2[1] - coord1[1]  # diferença leste-oeste
                        
                        # Turbina j deve estar ao sul de turbina i NA MESMA COLUNA
                        if dx > 0 and dy == 0:  # mesma coluna (dy = 0)
                            distance = abs(dx)
                            penalty = max(0, self.max_penalty - distance * self.decay_factor)
                            if penalty > 0:
                                penalties[(i, j)] = penalty
        
        return penalties

def generate_csv_filename(config):
    """Gera nome do arquivo CSV baseado na configuração"""
    # Extrair informações da configuração
    grid_size = f"{config['grid']['rows']}x{config['grid']['cols']}"
    
    # Direção do vento
    wind_dir = config["wind"]["direction"]
    if wind_dir == [0, 1]:
        wind_suffix = "oeste_leste"
    elif wind_dir == [1, 0]:
        wind_suffix = "norte_sul"
    else:
        wind_suffix = f"dir_{wind_dir[0]}_{wind_dir[1]}"
    
    # Modo do score
    score_mode = config["score"]["mode"]
    
    # Restrições
    constraints = config.get("constraints", {})
    enforce_constraints = constraints.get("enforce_constraints", False)
    constraints_suffix = "com_restricoes" if enforce_constraints else "sem_restricoes"
    
    # Nome do arquivo
    filename = f"qaoa_resultados_{grid_size}_{wind_suffix}_{score_mode}_{constraints_suffix}.csv"
    return filename

def log_simulation_results(config, config_file, optimal_value, best_bitstring, cost_history, counts, execution_time):
    """Registra os resultados da simulação no arquivo CSV apropriado"""
    
    # Criar pasta de resultados se não existir
    results_dir = "resultados_simulacoes"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 Pasta {results_dir} criada")
    
    # Gerar nome do arquivo CSV
    csv_filename = generate_csv_filename(config)
    csv_path = os.path.join(results_dir, csv_filename)
    
    # Verificar se arquivo existe para decidir se adiciona cabeçalho
    file_exists = os.path.exists(csv_path)
    
    # Extrair parâmetros para logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Parâmetros do grid e wind
    rows = config["grid"]["rows"]
    cols = config["grid"]["cols"]
    wind_direction = config["wind"]["direction"]
    
    # Wake effects/penalties
    if "wake_effects" in config:
        base_penalty = config["wake_effects"]["base_penalty"]
        distance_decay = config["wake_effects"]["distance_decay"]
    else:
        base_penalty = config["penalties"]["max_penalty"]
        distance_decay = config["penalties"]["decay_factor"]
    
    # Score parameters
    score_config = config["score"]
    score_mode = score_config["mode"]
    uniform_value = score_config.get("uniform_value", None) if score_mode == "uniform" else None
    
    # QAOA parameters
    qaoa_config = config["qaoa"]
    layers = qaoa_config["layers"]
    optimizer_name = qaoa_config["optimizer"]
    maxiter = qaoa_config.get("optimizer_options", {}).get("maxiter", None)
    rhobeg = qaoa_config.get("optimizer_options", {}).get("rhobeg", None)
    shots = qaoa_config["shots"]
    gamma_range_str = qaoa_config.get("gamma_range", qaoa_config.get("initial_param_range", "2*pi"))
    beta_range_str = qaoa_config.get("beta_range", qaoa_config.get("initial_param_range", "pi"))
    
    # Constraints parameters
    constraints = config.get("constraints", {})
    enforce_constraints = constraints.get("enforce_constraints", False)
    min_turbines = constraints.get("min_turbines", None)
    max_turbines = constraints.get("max_turbines", None)
    constraint_penalty = constraints.get("constraint_penalty", None)
    
    # Solution analysis
    solution = bitstring_to_grid(best_bitstring)
    num_turbines = sum(solution)
    best_count = counts[best_bitstring]
    best_probability = best_count / sum(counts.values())
    
    # Calcular scores
    optimizer_obj = QAOATurbineOptimizer(config_file)
    total_score = sum(solution[i] * optimizer_obj.score[i] for i in range(len(solution)))
    
    # Penalidades de esteira ativas e contagem de turbinas com penalty
    total_wake_penalty = 0
    turbines_with_penalty = set()  # Usar set para evitar duplicatas
    
    for (i, j), penalty in optimizer_obj.wake_penalties.items():
        if solution[i] == 1 and solution[j] == 1:
            total_wake_penalty += penalty
            # Apenas turbina downstream (j) sofre efeito da esteira
            turbines_with_penalty.add(j)
    
    num_turbines_with_penalty = len(turbines_with_penalty)
    
    # Calcular relação turbinas com penalty / turbinas totais
    penalty_ratio = num_turbines_with_penalty / num_turbines if num_turbines > 0 else 0
    
    net_score = total_score - total_wake_penalty
    
    # Preparar linha de dados
    row_data = [
        timestamp,
        config_file,
        f"{rows}x{cols}",
        f"{wind_direction[0]}-{wind_direction[1]}",
        base_penalty,
        distance_decay,
        score_mode,
        uniform_value if uniform_value is not None else "",
        layers,
        optimizer_name,
        maxiter if maxiter is not None else "",
        rhobeg if rhobeg is not None else "",
        shots,
        gamma_range_str,
        beta_range_str,
        enforce_constraints,
        min_turbines if min_turbines is not None else "",
        max_turbines if max_turbines is not None else "",
        constraint_penalty if constraint_penalty is not None else "",
        best_bitstring,
        f"{best_probability:.6f}",
        optimal_value,
        total_score,
        total_wake_penalty,
        net_score,
        len(cost_history),
        f"{execution_time:.2f}",
        min(cost_history) if cost_history else "",
        max(cost_history) if cost_history else "",
        cost_history[-1] if cost_history else "",
        num_turbines,
        num_turbines_with_penalty,
        f"{penalty_ratio:.6f}"
    ]
    
    # Cabeçalho do CSV
    header = [
        "timestamp",
        "config_file", 
        "grid_size",
        "wind_direction",
        "base_penalty",
        "distance_decay", 
        "score_mode",
        "uniform_value",
        "layers",
        "optimizer",
        "maxiter",
        "rhobeg", 
        "shots",
        "gamma_range",
        "beta_range", 
        "enforce_constraints",
        "min_turbines",
        "max_turbines", 
        "constraint_penalty",
        "best_bitstring",
        "best_probability",
        "optimal_value",
        "total_score",
        "wake_penalty",
        "net_score",
        "iterations", 
        "execution_time_s",
        "min_cost",
        "max_cost",
        "final_cost",
        "num_turbines",
        "turbines_with_penalty",
        "penalty_ratio"
    ]
    
    # Escrever no arquivo CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Adicionar cabeçalho se arquivo novo
        if not file_exists:
            writer.writerow(header)
            print(f"📝 Novo arquivo CSV criado: {csv_path}")
        
        # Adicionar dados
        writer.writerow(row_data)
        print(f"✅ Resultados salvos em: {csv_path}")
        print(f"   • Timestamp: {timestamp}")
        print(f"   • Configuração: {config_file}")
        print(f"   • Score líquido: {net_score:.2f}")
        print(f"   • Turbinas com penalty: {num_turbines_with_penalty}/{num_turbines} ({penalty_ratio:.1%})")

def run_optimization(optimizer):
    """Executa a otimização QAOA"""
    # Definir variáveis globais para compatibilidade
    global score, positions_coords, wake_penalties
    score = optimizer.score
    positions_coords = optimizer.positions_coords
    wake_penalties = optimizer.wake_penalties

    print(f"\nMATRIZ DE INTERFERÊNCIA (penalidades SE ambas turbinas estiverem instaladas):")
    wind_desc = "Oeste → Leste" if optimizer.wind_direction == (0, 1) else "Norte → Sul"
    print(f"Direção do vento: {wind_desc}")
    print()

    # Mostrar de forma mais clara com coordenadas
    for (i, j), penalty in wake_penalties.items():
        coord1 = positions_coords[i]
        coord2 = positions_coords[j]
        print(f"  Posição {coord1} → Posição {coord2}: penalidade {penalty}")
        
    print(f"\nTotal de {len(wake_penalties)} possíveis interferências no grid {optimizer.rows}x{optimizer.cols}")

    # Executar QAOA
    try:
        # Obter maxiter das opções do otimizador ou usar padrão
        max_iter = optimizer.config["qaoa"].get("optimizer_options", {}).get("maxiter", 50)
        counts, optimal_value, cost_history, execution_time, param_hist = run_qaoa(p=optimizer.config["qaoa"]["layers"], max_iter=max_iter, use_ibm_quantum=False, args=None, config_file=config_file)
        
        # Gerar gráfico de evolução do custo (condicionado)
        if PLOT_ENABLED:
            plot_cost_evolution(cost_history, config_file, rhobeg=param_hist.get('rhobeg'))
        
        analyze_results(counts)
        
        # Trajetória γ-β (condicionado)
        if PLOT_ENABLED:
            try:
                from utils import plot_gamma_beta_trajectory as _plot_traj
                _plot_traj(param_hist.get("gamma_history", []), param_hist.get("beta_history", []), config_file, rhobeg=param_hist.get('rhobeg'))
            except Exception:
                pass
        
    except Exception as e:
        print(f"❌ ERRO DETALHADO NA EXECUÇÃO DO QAOA:")
        print(f"   • Tipo do erro: {type(e).__name__}")
        print(f"   • Mensagem: {str(e)}")
        
        # Mostrar traceback completo para debug
        import traceback
        print(f"   • Stack trace:")
        traceback.print_exc()
        
        print("\n💔 Execução interrompida devido ao erro.")
        sys.exit(1)

# Instanciar o otimizador com arquivo apropriado
config_file = get_config_file()
optimizer = QAOATurbineOptimizer(config_file)

# Exibir matriz de interferências se configurado
if optimizer.config.get("display", {}).get("show_interference_matrix", True):
    display_interference_matrix(optimizer)
else:
    print("📊 Matriz de interferências: OCULTA (configurado no JSON)")
    print(f"Total de {len(optimizer.wake_penalties)} possíveis interferências no grid {optimizer.rows}x{optimizer.cols}")

# Manter compatibilidade com código existente
score = optimizer.score
positions_coords = optimizer.positions_coords
wake_penalties = optimizer.wake_penalties

def create_cost_hamiltonian():
    """
    Cria o Hamiltoniano de custo para QAOA.
    
    Objetivo: Maximizar (score - wake_penalties) = Minimizar -(score - wake_penalties)
    
    PROBLEMA IDENTIFICADO: O termo ZZ penaliza estados incorretos:
    - |00⟩: ZZ = +1 → penaliza (sem turbinas, não deveria penalizar)
    - |01⟩: ZZ = -1 → recompensa (uma turbina, não deveria recompensar) 
    - |10⟩: ZZ = -1 → recompensa (uma turbina, não deveria recompensar)
    - |11⟩: ZZ = +1 → penaliza (ambas turbinas, CORRETO)
    
    SOLUÇÃO: Reformular para penalizar apenas |11⟩ = turbinas i e j ambas presentes.
    """
    pauli_list = []
    const_offset = 0.0  # OTIMIZAÇÃO: Acumular termos I para reduzir overhead
    
    # Termos lineares (score): -score[i] * Z[i] 
    for i in range(optimizer.n_positions):
        print(i,":",score[i])
        # |0⟩: Z = +1 → penaliza
        # |1⟩: Z = -1 → recompensa 
        # |any⟩: -I = -1 → recompensa
        pauli_list.append(("Z", [i], +score[i]/2)) #|0⟩:+score/2, |1⟩:-score/2 , negativo para recompensar
        const_offset += -score[i]/2 #termo global negativo para recompensar (acumulado)
    
    # OPÇÃO 1: Correção completa para penalizar apenas |11⟩
    # Para cada par de turbinas com wake interference:
    
    for (i, j), wake_penalty in wake_penalties.items():
        print(i,",",j,":",wake_penalty)
              
        pauli_list.append(("ZZ", [i, j], wake_penalty/4))  # penaliza |00⟩ e |11⟩   
        pauli_list.append(("Z", [i], -wake_penalty/4))     # penaliza |1*⟩   Z1|0*⟩=+|0*⟩   Z1|1*⟩=-|1*⟩, por issoo sinal de menos
        pauli_list.append(("Z", [j], -wake_penalty/4))     # penaliza |*1⟩
        const_offset += wake_penalty/4  # termo global positivo para penalizar (acumulado)
    
    # Adicionar único termo I com offset total
    if abs(const_offset) > 1e-10:  # Evitar termo zero
        pauli_list.append(("I", [], const_offset))
        print(f"Termo constante total acumulado: {const_offset:.3f}")
    
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)

def params_array_to_dict(params_array, ansatz):
    """Converte array de parâmetros para dicionário usando ordem do circuito"""
    circuit_params = list(ansatz.parameters)
    return {param: params_array[i] for i, param in enumerate(circuit_params)}

def objective_function_with_constraints(params, estimator, ansatz, cost_hamiltonian, simple_sampler):
    """Função objetivo que inclui penalty dinâmica para restrições min/max turbinas"""
    
    # 1. Calcula energia do Hamiltoniano (QAOA normal)
    # CORREÇÃO CRÍTICA: Usar binding por dicionário no Estimator
    param_dict = params_array_to_dict(params, ansatz)
    job = estimator.run([(ansatz, cost_hamiltonian, param_dict)])
    result = job.result()
    qaoa_energy = result[0].data.evs  # Valor de expectativa (já é escalar)
    
    # 2. Se restrições não estão ativas, retorna energia normal
    if not optimizer.enforce_constraints:
        return qaoa_energy
    
    # 3. Extrai distribuição completa para calcular penalty esperada
    quasi_dist = simple_sampler(ansatz, params)
       
    # 4. Calcula penalty esperada sobre TODA a distribuição (mais estável)
    expected_penalty = 0
    
    for config, probability in quasi_dist.items():
        # Conta turbinas nesta configuração
        num_turbines = bin(config).count('1')        
        # Calcula penalty para esta configuração
        config_penalty = 0
        if num_turbines < optimizer.min_turbines:
            violation = optimizer.min_turbines - num_turbines
            config_penalty = optimizer.constraint_penalty * violation**2
        elif num_turbines > optimizer.max_turbines:
            violation = num_turbines - optimizer.max_turbines
            config_penalty = optimizer.constraint_penalty * violation**2
        
        # Adiciona penalty ponderada pela probabilidade
        expected_penalty += probability * config_penalty
        
    #print(qaoa_energy)
    #print(expected_penalty)
    
    # 5. Retorna energia + penalty esperada (minimização)
    return qaoa_energy + expected_penalty

# Importar time para benchmarking
import time
    
# 4. FUNÇÃO AUXILIAR PARA VALIDAR RESTRIÇÕES

def create_qaoa_ansatz(cost_hamiltonian, p):
    """Cria o ansatz QAOA manualmente para compatibilidade com EstimatorV2"""
    n_qubits = cost_hamiltonian.num_qubits
    
    # Criar circuito
    circuit = QuantumCircuit(n_qubits)
    
    # Estado inicial: superposição uniforme
    for i in range(n_qubits):
        circuit.h(i)
    
    # Parâmetros QAOA
    gammas = [Parameter(f'gamma_{i}') for i in range(p)]
    betas = [Parameter(f'beta_{i}') for i in range(p)]
    
    # Camadas QAOA
    for layer in range(p):
        # Aplicar operador de custo (problema-dependente)
        # Para cada termo no Hamiltoniano
        for pauli, coeff in cost_hamiltonian.to_list():
            #print(pauli)
            #print(coeff)
            # CORREÇÃO: Verificar ZZ primeiro, depois Z
            if pauli.count('Z') == 2:  # Termos ZZ (dois qubits)
                qubits = [i for i, op in enumerate(pauli) if op == 'Z']
                if len(qubits) == 2:
                    circuit.rzz(2 * gammas[layer] * coeff.real, qubits[0], qubits[1])
            elif 'Z' in pauli:  # Termos Z (single qubit)
                for i, op in enumerate(pauli):
                    if op == 'Z':
                        circuit.rz(2 * gammas[layer] * coeff.real, i)
        
        # Aplicar operador de mistura (X rotations)
        for i in range(n_qubits):
            circuit.rx(2 * betas[layer], i)
    
    return circuit

  



def run_qaoa(p, max_iter=50, use_ibm_quantum=False, args=None, config_file=None):
    """Executa o algoritmo QAOA usando APIs modernas"""
    print(f"\nConfigurando QAOA com p={p} camadas...")
    
    # Marcar início do tempo de execução
    start_time = time.time()
    
    # Criar Hamiltoniano e ansatz
    cost_hamiltonian = create_cost_hamiltonian()
    ansatz = create_qaoa_ansatz(cost_hamiltonian, p)
    
    print(f"Ansatz criado com {ansatz.num_qubits} qubits e {len(ansatz.parameters)} parâmetros")
    
    # Plotar circuito quântico se solicitado
    if args and getattr(args, 'plot_quantum', False):
        plot_quantum_circuit(ansatz, config_file or 'config.json')
    
    # Configurar EstimatorV2 baseado no modo
    shots = optimizer.config.get("qaoa", {}).get("shots", 1024)
    
    if use_ibm_quantum:
        try:
            from qiskit_ibm_runtime import EstimatorV2 as IBMEstimatorV2, QiskitRuntimeService
            
            # Carregar configurações IBM
            api_key = load_ibm_api_key()
            ibm_config = load_ibm_config()
            
            # Usar configurações do ibm.json
            instance = ibm_config['instance']
            primary_backend = ibm_config['backends']['primary']
            fallback_backend = ibm_config['backends']['fallback']
            
            # Salvar credenciais com instância
            QiskitRuntimeService.save_account(
                token=api_key, 
                instance=instance,
                overwrite=True
            )
            
            # Conectar especificando a instância
            service = QiskitRuntimeService(instance=instance)
            
            # Tentar backend primário primeiro, depois fallback
            try:
                backend = service.backend(primary_backend)
                print(f"   🎯 Usando backend primário: {primary_backend}")
            except:
                backend = service.backend(fallback_backend)
                print(f"   🔄 Fallback para: {fallback_backend}")
            # Criar EstimatorV2 com configurações de shots
            estimator = IBMEstimatorV2(mode=backend, options={"default_shots": shots})
            
            print(f"🌐 Conectado ao IBM Quantum: {backend.name}")
            print(f"   • Qubits disponíveis: {backend.configuration().num_qubits}")
            print(f"   • Status: {backend.status().status_msg}")
            print(f"   • Shots por iteração: {shots}")
            print(f"   • Transpilação automática: nível 3")
            
        except ImportError:
            print("❌ qiskit-ibm-runtime não encontrado. Execute: pip install qiskit-ibm-runtime")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Erro ao conectar com IBM Quantum: {e}")
            sys.exit(1)
    else:
        # Modo simulação local
        estimator = EstimatorV2()
        estimator.options.default_shots = shots
        
        # Configurar número de threads do AER se especificado
        aer_config = optimizer.config.get("aer", {})
        max_threads = aer_config.get("max_parallel_threads", None)
        
        if max_threads is not None:
            estimator.options.backend_options = {"max_parallel_threads": max_threads}
            print(f"🖥️  Usando simulação local com {shots} shots e {max_threads} threads")
        else:
            print(f"🖥️  Usando simulação local com {shots} shots (threads automático)")
    
    # Para IBM Quantum, usar transpilação manual recomendada pela IBM
    if use_ibm_quantum:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        
        # Usar nível de otimização do ibm.json
        opt_level = ibm_config.get('transpilation', {}).get('optimization_level', 3)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
        ansatz = pm.run(ansatz)
        print(f"   ⚙️  Transpilação: nível {opt_level}")
        
        # Expandir o Hamiltoniano para match com o número de qubits do circuito transpilado
        n_qubits_transpiled = ansatz.num_qubits
        if n_qubits_transpiled > cost_hamiltonian.num_qubits:
            # Criar Hamiltoniano expandido corretamente
            extra_qubits = n_qubits_transpiled - cost_hamiltonian.num_qubits
            identity_op = SparsePauliOp.from_list([("I" * extra_qubits, 1.0)])
            cost_hamiltonian = cost_hamiltonian.tensor(identity_op)
            print(f"   • Hamiltoniano expandido de {optimizer.n_positions} para {n_qubits_transpiled} qubits")
        
        print(f"✅ Circuito transpilado para {backend.name} usando preset pass manager")
        
    # Contador para acompanhar iterações e histórico de custos
    iteration_count = [0]  # Lista para permitir modificação dentro da função aninhada
    cost_history = []  # Lista para armazenar histórico de custos
    # Histórico γ/β por camada (preparado após detectar ordem dos params)
    gamma_indices = []
    beta_indices = []
    gamma_history = None
    beta_history = None
    
    def cost_function(params):
        """Função de custo com penalty dinâmica para restrições"""
        iteration_count[0] += 1
        
        # Usar nossa nova função com penalidades dinâmicas
        # Criar sampler simples baseado no simulador existente
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        def simple_sampler(ansatz, params):
            # CORREÇÃO: Usar binding por dicionário para garantir ordem correta
            # Converter array para dicionário usando ordem explícita do circuito
            param_dict = params_array_to_dict(params, ansatz)
            circuit = ansatz.assign_parameters(param_dict)
            circuit.measure_all()
            
            if use_ibm_quantum:
                # Usar IBM Quantum
                from qiskit import transpile
                from qiskit_ibm_runtime import SamplerV2
                
                sampler = SamplerV2(mode=backend)
                opt_level = ibm_config.get('transpilation', {}).get('optimization_level', 3)
                transpiled = transpile(circuit, backend, optimization_level=opt_level)
                job = sampler.run([transpiled], shots=shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                # Simular localmente com configuração de threads
                aer_config = optimizer.config.get("aer", {})
                max_threads = aer_config.get("max_parallel_threads", None)
                
                if max_threads is not None:
                    simulator = AerSimulator(max_parallel_threads=max_threads)
                else:
                    simulator = AerSimulator()
                    
                transpiled = transpile(circuit, simulator)
                job = simulator.run(transpiled, shots=shots)
                counts = job.result().get_counts()
            
            # Converter para quasi_dist format
            total_shots = sum(counts.values())
            quasi_dist = {int(bitstring, 2): count/total_shots for bitstring, count in counts.items()}
            return quasi_dist
        
        cost_value = objective_function_with_constraints(params, estimator, ansatz, cost_hamiltonian, simple_sampler)
        
        # Formatar parâmetros para exibição (limitando a 6 parâmetros para não poluir)
        if len(params) <= 6:
            params_str = ", ".join([f"{p:6.3f}" for p in params])
        else:
            # Mostrar apenas os primeiros 3 e últimos 3
            params_str = ", ".join([f"{p:6.3f}" for p in params[:3]]) + " ... " + ", ".join([f"{p:6.3f}" for p in params[-3:]])
        
        # Armazenar histórico para plotagem
        cost_history.append(cost_value)
        # Capturar trajetória γ/β por camada nesta avaliação
        try:
            for l in range(n_layers):
                g_idx = gamma_indices[l] if l < len(gamma_indices) else -1
                b_idx = beta_indices[l] if l < len(beta_indices) else -1
                if g_idx != -1 and b_idx != -1 and g_idx < len(params) and b_idx < len(params):
                    gamma_history[l].append(float(params[g_idx]))
                    beta_history[l].append(float(params[b_idx]))
        except Exception:
            pass
        
        # Imprimir progresso da otimização
        print(f"  Iteração {iteration_count[0]:3d}: Custo = {cost_value:8.4f} | Params: [{params_str}]")
        
        return cost_value
    
    # Otimização clássica dos parâmetros
    optimizer_method = optimizer.config["qaoa"]["optimizer"]
    print(f"Iniciando otimização dos parâmetros usando {optimizer_method}...")
    
    # Chute inicial: ranges físicos separados para γ e β
    n_layers = optimizer.config["qaoa"]["layers"]
    
    # Ranges separados (backward compatibility)
    qaoa_config = optimizer.config["qaoa"]
    
    # Gamma range (cost parameters)
    if "gamma_range" in qaoa_config:
        gamma_range_str = qaoa_config["gamma_range"]
    else:
        # Fallback: usar initial_param_range ou padrão físico
        gamma_range_str = qaoa_config.get("initial_param_range", "2*pi")
    
    # Beta range (mixing parameters)  
    if "beta_range" in qaoa_config:
        beta_range_str = qaoa_config["beta_range"]
    else:
        # Fallback: usar initial_param_range ou padrão físico
        beta_range_str = qaoa_config.get("initial_param_range", "pi")
    
    # Converter strings para valores numéricos
    gamma_range = eval(gamma_range_str.replace("pi", "np.pi"))
    beta_range = eval(beta_range_str.replace("pi", "np.pi"))
    
    # CORREÇÃO CRÍTICA: Usar binding por dicionário para evitar problemas de ordem
    # Extrair parâmetros explícitos do ansatz
    circuit_params = list(ansatz.parameters)
    
    # Identificar gammas e betas do circuito
    gammas = [p for p in circuit_params if 'gamma' in str(p)]
    betas = [p for p in circuit_params if 'beta' in str(p)]
    
    # Ordenar para garantir sequência correta
    gammas.sort(key=lambda p: int(str(p).split('_')[1]))
    betas.sort(key=lambda p: int(str(p).split('_')[1])) 
    
    print(f"Parâmetros detectados no circuito:")
    print(f"  • Gammas: {[str(g) for g in gammas]}")
    print(f"  • Betas: {[str(b) for b in betas]}")
    
    print(f"Ranges configurados:")
    print(f"  • γ range: [0, {gamma_range:.3f}] = [0, {gamma_range/np.pi:.2f}π]")
    print(f"  • β range: [0, {beta_range:.3f}] = [0, {beta_range/np.pi:.2f}π]")
    print(f"Valores iniciais (ordem do circuito):")
    
    # CORREÇÃO: Gerar initial_params na ordem EXATA do circuito (robusto)
    initial_params = []
    for param in circuit_params:
        if 'gamma' in str(param):
            # Parâmetro γ: usar gamma_range
            val = np.random.uniform(0, gamma_range)
            initial_params.append(val)
            print(f"    {param}: {val:.3f} (γ range)")
        elif 'beta' in str(param):
            # Parâmetro β: usar beta_range  
            val = np.random.uniform(0, beta_range)
            initial_params.append(val)
            print(f"    {param}: {val:.3f} (β range)")
        else:
            print(f"    ❌ UNKNOWN PARAM: {param}")
    
    initial_params = np.array(initial_params)
    print(f"Array final (ordem do circuito): {[f'{p:.3f}' for p in initial_params]}")

    # Mapear índices de γ_l e β_l na ordem do vetor de parâmetros (ordem do circuito)
    gamma_indices = [-1] * n_layers
    beta_indices = [-1] * n_layers
    for idx, param in enumerate(circuit_params):
        name = str(param)
        if 'gamma_' in name:
            l = int(name.split('_')[1])
            if 0 <= l < n_layers:
                gamma_indices[l] = idx
        elif 'beta_' in name:
            l = int(name.split('_')[1])
            if 0 <= l < n_layers:
                beta_indices[l] = idx
    # Preparar histórico dos parâmetros
    gamma_history = [[] for _ in range(n_layers)]
    beta_history = [[] for _ in range(n_layers)]
    
    # Otimizar usando algoritmo configurado
    optimizer_method = optimizer.config["qaoa"]["optimizer"]
    
    # Carregar opções específicas do otimizador do config
    base_options = {}
    if "optimizer_options" in optimizer.config["qaoa"]:
        base_options = optimizer.config["qaoa"]["optimizer_options"].copy()
        print(f"Usando opções personalizadas: {base_options}")
    
    # Aplicar opções padrão se não especificadas
    if optimizer_method in ['L-BFGS-B', 'BFGS']:
        # Para algoritmos baseados em gradiente, usar maxfun como padrão
        if 'maxfun' not in base_options and 'maxiter' not in base_options:
            base_options['maxfun'] = max_iter
    else:
        # Para outros algoritmos, usar maxiter como padrão  
        if 'maxiter' not in base_options:
            base_options['maxiter'] = max_iter
    
    print(f"Opções finais do otimizador: {base_options}")

    # Capturar rhobeg efetivo (se aplicável)
    rhobeg_used = None
    if optimizer_method.upper() == 'COBYLA':
        rhobeg_used = base_options.get('rhobeg', None)
    
    result = minimize(cost_function, initial_params, method=optimizer_method, 
                     options=base_options)
    
    print(f"Otimização concluída em {result.nfev} avaliações")
    print(f"Valor ótimo encontrado: {result.fun}")
    
    # Executar circuito final para obter distribuição
    # CORREÇÃO FINAL: Converter vetor ótimo para dicionário antes de bindar
    optimal_param_dict = params_array_to_dict(result.x, ansatz)
    print(f"Parâmetros ótimos encontrados:")
    for param, val in optimal_param_dict.items():
        param_type = "γ" if 'gamma' in str(param) else "β"
        print(f"  {param}: {val:.3f} ({param_type})")
    
    final_circuit = ansatz.assign_parameters(optimal_param_dict)
    final_circuit.measure_all()
    
    if use_ibm_quantum:
        # Executar no hardware IBM
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2
        
        sampler = SamplerV2(mode=backend)
        opt_level = ibm_config.get('transpilation', {}).get('optimization_level', 3)
        transpiled_circuit = transpile(final_circuit, backend, optimization_level=opt_level)
        
        print(f"🚀 Executando circuito final no {backend.name}...")
        job = sampler.run([transpiled_circuit], shots=shots)
        result_final = job.result()
        counts = result_final[0].data.meas.get_counts()
        
    else:
        # Simular localmente
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        # Configurar número de threads do AER se especificado
        aer_config = optimizer.config.get("aer", {})
        max_threads = aer_config.get("max_parallel_threads", None)
        
        if max_threads is not None:
            simulator = AerSimulator(max_parallel_threads=max_threads)
        else:
            simulator = AerSimulator()
            
        transpiled_circuit = transpile(final_circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        counts = job.result().get_counts()
    
    # Calcular tempo total de execução
    execution_time = time.time() - start_time
    
    return counts, result.fun, cost_history, execution_time, {"gamma_history": gamma_history, "beta_history": beta_history, "rhobeg": rhobeg_used}



# 6. FUNÇÃO analyze_results COM VALIDAÇÃO DE RESTRIÇÕES E RANKING
def analyze_results(counts):
    """Analisa os resultados do QAOA com validação de restrições"""
    print("\n" + "="*50)
    print("RESULTADOS DO QAOA")
    print("="*50)
    
    # Encontrar a melhor solução
    best_bitstring = max(counts, key=counts.get)
    best_count = counts[best_bitstring]
    best_probability = best_count / sum(counts.values())
    
    print(f"\nMelhor solução encontrada: {best_bitstring}")
    print(f"Probabilidade: {best_probability:.3f} ({best_count}/1024 medições)")
    
    # Converter para formato de grid
    solution = bitstring_to_grid(best_bitstring)
    display_grid(solution, optimizer, "MELHOR CONFIGURAÇÃO")
    
    # NOVO: Validar restrições de forma destacada
    violations = validate_constraints(solution, optimizer)
    print(f"\n🎯 VERIFICAÇÃO DE RESTRIÇÕES:")
    for violation in violations:
        print(f"   {violation}")
    
    # Mostrar penalidades efetivas
    total_penalty = show_active_penalties(solution, wake_penalties, positions_coords)
    
    # Métricas detalhadas
    installed_positions = [i for i in range(optimizer.n_positions) if solution[i] == 1]
    total_score = sum(solution[i] * score[i] for i in range(optimizer.n_positions))
    num_turbines = len(installed_positions)
    
    print(f"\n📊 MÉTRICAS DETALHADAS:")
    print(f"   • Turbinas instaladas: {num_turbines}")
    
    # Mostrar status das restrições
    if optimizer.enforce_constraints:
        print(f"   • Restrição mínima: {optimizer.min_turbines} (diferença: {num_turbines - optimizer.min_turbines:+d})")
        print(f"   • Restrição máxima: {optimizer.max_turbines} (diferença: {optimizer.max_turbines - num_turbines:+d})")
        
        # Calcular penalidade por violação de restrições
        constraint_penalty = 0
        if num_turbines < optimizer.min_turbines:
            constraint_penalty += optimizer.constraint_penalty * (optimizer.min_turbines - num_turbines)
        if num_turbines > optimizer.max_turbines:
            constraint_penalty += optimizer.constraint_penalty * (num_turbines - optimizer.max_turbines)
        
        if constraint_penalty > 0:
            print(f"   • Penalidade por restrições: {constraint_penalty}")
    else:
        print(f"   • Sem restrições de número")
    
    print(f"   • Score total: {total_score:.2f}")
    print(f"   • Penalidades de esteira: {total_penalty:.2f}")
    net_score = total_score - total_penalty
    if optimizer.enforce_constraints and constraint_penalty > 0:
        net_score -= constraint_penalty
        print(f"   • Score líquido final: {net_score:.2f} (descontando restrições)")
    else:
        print(f"   • Score líquido: {net_score:.2f}")
    
    # Mostrar produção por posição
    print(f"\n⚡ PRODUÇÃO POR POSIÇÃO:")
    for i in range(optimizer.n_positions):
        row = i // optimizer.cols
        col = i % optimizer.cols
        status = "🌪️" if solution[i] == 1 else "⬜"
        sc = score[i] if solution[i] == 1 else 0
        print(f"   Pos ({row},{col}): {status} Score: {sc:.1f}")
        
    # Top 5 soluções com visualização
    print(f"\n🏆 TOP 5 SOLUÇÕES MAIS PROVÁVEIS:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for i, (bitstring, count) in enumerate(sorted_counts[:5]):
        prob = count / sum(counts.values())
        value = evaluate_solution(bitstring, score, wake_penalties, optimizer)
        sol = bitstring_to_grid(bitstring)
        
        # Formato compacto do grid
        if optimizer.rows == 2 and optimizer.cols == 3:
            grid_compact = f"[{sol[0]}{sol[1]}{sol[2]}|{sol[3]}{sol[4]}{sol[5]}]"
        else:
            # Para grids maiores, mostrar apenas as posições ativas
            active_pos = [str(i) for i in range(optimizer.n_positions) if sol[i] == 1]
            grid_compact = f"[{','.join(active_pos) if active_pos else 'vazio'}]"
        
        print(f"   {i+1}. {grid_compact} - {prob:.3f} - Score: {value:.2f}")
        
        # Mostrar grid completo apenas para as 3 melhores
        if i < 3:
            display_grid(sol, optimizer, f"Solução #{i+1}")
            
    # Resumo final com destaque para restrições
    print(f"\n" + "="*50)
    print(f"RESUMO FINAL")
    print(f"="*50)
    if optimizer.enforce_constraints:
        if num_turbines < optimizer.min_turbines or num_turbines > optimizer.max_turbines:
            print(f"⚠️  ATENÇÃO: Solução VIOLA restrições!")
            print(f"   Exigido: [{optimizer.min_turbines}, {optimizer.max_turbines}] turbinas")
            print(f"   Encontrado: {num_turbines} turbinas")
        else:
            print(f"✅ Solução respeita todas as restrições!")
            print(f"   {num_turbines} turbinas dentro do intervalo [{optimizer.min_turbines}, {optimizer.max_turbines}]")
    else:
        print(f"ℹ️  Execução sem restrições de número de turbinas")
    print(f"🏆 Score líquido final: {net_score:.2f}")


# Executar a otimização apenas se executado diretamente
if __name__ == "__main__":
    # Primeiro processar argumentos
    args = parse_arguments()
    # Ajustar flag global para funções utilitárias
    PLOT_ENABLED = bool(getattr(args, 'plot', False))
    
    # Se solicitado, listar configurações e sair
    if args.list_configs:
        list_available_configs()
        sys.exit(0)
    
    # Inicializar com configuração especificada
    optimizer = QAOATurbineOptimizer(args.config)
    score = optimizer.score
    wake_penalties = optimizer.wake_penalties
    
    # Se solicitado benchmark, executar e sair
    if args.benchmark_hamiltonian:
        compare_hamiltonian_implementations()
        sys.exit(0)
    
    # Verificar se IBM Quantum foi solicitado
    use_ibm = args.ibm_quantum
    if use_ibm:
        # Confirmar execução no IBM Quantum
        if not confirm_ibm_execution(optimizer.config):
            sys.exit(0)
    
    print(f"\n🚀 INICIANDO OTIMIZAÇÃO QAOA...")
    print(f"\n📋 RESUMO DAS RESTRIÇÕES:")
    if optimizer.enforce_constraints:
        print(f"   🎯 Número EXATO de turbinas requerido: {optimizer.min_turbines}")
        print(f"   ⚠️  Violações serão penalizadas com fator {optimizer.constraint_penalty}")
        print(f"   🔍 O QAOA deve encontrar exatamente {optimizer.min_turbines} turbinas")
    else:
        print(f"   🆓 SEM restrições - qualquer número de turbinas é válido")
        print(f"   📊 Intervalo permitido: 0 a {optimizer.n_positions} turbinas")
    
    try:
        # Usar parâmetros QAOA da configuração
        p_layers = optimizer.config["qaoa"]["layers"]
        max_iterations = optimizer.config["qaoa"].get("optimizer_options", {}).get("maxiter", 50)
        print(f"\n🔧 QAOA: {p_layers} camadas, {max_iterations} iterações do otimizador")
        
        # Executar QAOA
        counts, optimal_value, cost_history, execution_time, param_hist = run_qaoa(p=p_layers, max_iter=max_iterations, use_ibm_quantum=use_ibm, args=args, config_file=args.config)
        
        # Gerar gráficos somente se --plot for passado
        if getattr(args, 'plot', False):
            cfg_name = optimizer.config.get('grid', {}).get('description', args.config)
            rhobeg_used = param_hist.get('rhobeg')
            plot_cost_evolution(cost_history, cfg_name, rhobeg=rhobeg_used)
            plot_gamma_beta_trajectory(param_hist.get('gamma_history', []), param_hist.get('beta_history', []), cfg_name, rhobeg=rhobeg_used)
            
            # Gerar visualização do grid mais provável
            best_bitstring = max(counts, key=counts.get)
            best_probability = counts[best_bitstring] / sum(counts.values())
            solution = bitstring_to_grid(best_bitstring)
            
            # Calcular scores para a visualização
            total_score = sum(solution[i] * optimizer.score[i] for i in range(optimizer.n_positions))
            wake_penalty = sum(solution[i] * solution[j] * penalty 
                             for (i, j), penalty in optimizer.wake_penalties.items())
            
            create_grid_visualization(solution, optimizer, args.config, 
                                    best_probability, total_score, wake_penalty)
        
        analyze_results(counts)
        
        # Log dos resultados
        best_bitstring = max(counts, key=counts.get)
        log_simulation_results(
            config=optimizer.config, 
            config_file=args.config, 
            optimal_value=optimal_value,
            best_bitstring=best_bitstring,
            cost_history=cost_history, 
            counts=counts,
            execution_time=execution_time
        )
        
    except Exception as e:
        print(f"❌ ERRO DETALHADO NA EXECUÇÃO DO QAOA:")
        print(f"   • Tipo do erro: {type(e).__name__}")
        print(f"   • Mensagem: {str(e)}")
        
        # Mostrar traceback completo para debug
        import traceback
        print(f"   • Stack trace:")
        traceback.print_exc()
        
        print("\n💔 Execução interrompida devido ao erro.")
        sys.exit(1)
