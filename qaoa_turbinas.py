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
from utils import (parse_arguments, list_available_configs, load_config, get_config_file,
                   validate_constraints, evaluate_solution, show_active_penalties, 
                   bitstring_to_grid, display_grid, display_interference_matrix, plot_cost_evolution,
                   load_ibm_api_key, load_ibm_config, confirm_ibm_execution)
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
        counts, optimal_value, cost_history = run_qaoa(p=optimizer.config["qaoa"]["layers"], max_iter=max_iter)
        
        # Gerar gráfico de evolução do custo
        plot_cost_evolution(cost_history, config_file)
        
        analyze_results(counts)
        
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

def create_cost_hamiltonian_ANTIGO():
    """Cria o Hamiltoniano de custo usando SparsePauliOp moderno"""
    pauli_list = []
    
    # Termos lineares (score): -score[i] * Z[i] 
    for i in range(optimizer.n_positions):
        pauli_list.append(("Z", [i], -score[i]))  # Negativo para maximizar
    
    # Termos quadráticos (penalidades): penalty * Z[i] * Z[j]
    for (i, j), penalty in wake_penalties.items():
        pauli_list.append(("ZZ", [i, j], penalty))  # Positivo para penalizar
    
    # Usando from_sparse_list (método moderno recomendado)
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)
    
def create_cost_hamiltonian():
    """Cria o Hamiltoniano de custo - apenas score e wake penalties"""
    pauli_list = []
    
    # Termos lineares (score): -score[i] * Z[i] 
    for i in range(optimizer.n_positions):
        pauli_list.append(("Z", [i], -score[i]))  # Negativo para maximizar
    
    # Termos quadráticos (penalidades de esteira): wake_penalty * Z[i] * Z[j]
    for (i, j), wake_penalty in wake_penalties.items():
        pauli_list.append(("ZZ", [i, j], wake_penalty))  # Positivo para penalizar
    
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)


def objective_function_with_constraints(params, estimator, ansatz, cost_hamiltonian, simple_sampler):
    """Função objetivo que inclui penalty dinâmica para restrições min/max turbinas"""
    
    # 1. Calcula energia do Hamiltoniano (QAOA normal)
    # Criar PUB (Primitive Unified Block) para EstimatorV2
    job = estimator.run([(ansatz, cost_hamiltonian, params)])
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
            if 'Z' in pauli:
                # Aplicar rotações Z baseadas nos termos do Hamiltoniano
                for i, op in enumerate(pauli):
                    if op == 'Z':
                        circuit.rz(2 * gammas[layer] * coeff.real, i)
            elif 'ZZ' in pauli or pauli.count('Z') == 2:
                # Termos de dois qubits
                qubits = [i for i, op in enumerate(pauli) if op == 'Z']
                if len(qubits) == 2:
                    circuit.rzz(2 * gammas[layer] * coeff.real, qubits[0], qubits[1])
        
        # Aplicar operador de mistura (X rotations)
        for i in range(n_qubits):
            circuit.rx(2 * betas[layer], i)
    
    return circuit

  



def run_qaoa(p, max_iter=50, use_ibm_quantum=False):
    """Executa o algoritmo QAOA usando APIs modernas"""
    print(f"\nConfigurando QAOA com p={p} camadas...")
    
    # Criar Hamiltoniano e ansatz
    cost_hamiltonian = create_cost_hamiltonian()
    ansatz = create_qaoa_ansatz(cost_hamiltonian, p)
    
    print(f"Ansatz criado com {ansatz.num_qubits} qubits e {len(ansatz.parameters)} parâmetros")
    
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
        print(f"🖥️  Usando simulação local com {shots} shots por iteração")
    
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
    
    def cost_function(params):
        """Função de custo com penalty dinâmica para restrições"""
        iteration_count[0] += 1
        
        # Usar nossa nova função com penalidades dinâmicas
        # Criar sampler simples baseado no simulador existente
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        def simple_sampler(ansatz, params):
            # Criar circuito com parâmetros
            circuit = ansatz.assign_parameters(params)
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
                # Simular localmente
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
        
        # Imprimir progresso da otimização
        print(f"  Iteração {iteration_count[0]:3d}: Custo = {cost_value:8.4f} | Params: [{params_str}]")
        
        return cost_value
    
    # Otimização clássica dos parâmetros
    optimizer_method = optimizer.config["qaoa"]["optimizer"]
    print(f"Iniciando otimização dos parâmetros usando {optimizer_method}...")
    
    # Chute inicial: valores maiores para melhor gradiente
    initial_params = np.random.uniform(0, np.pi/4, len(ansatz.parameters))
    print(f"Parâmetros iniciais: {[f'{p:.3f}' for p in initial_params]}")
    
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
    
    result = minimize(cost_function, initial_params, method=optimizer_method, 
                     options=base_options)
    
    print(f"Otimização concluída em {result.nfev} avaliações")
    print(f"Valor ótimo encontrado: {result.fun}")
    
    # Executar circuito final para obter distribuição
    final_circuit = ansatz.assign_parameters(result.x)
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
        
        simulator = AerSimulator()
        transpiled_circuit = transpile(final_circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        counts = job.result().get_counts()
    
    return counts, result.fun, cost_history



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
        counts, optimal_value, cost_history = run_qaoa(p=p_layers, max_iter=max_iterations, use_ibm_quantum=use_ibm)
        
        # Gerar gráfico de evolução do custo
        plot_cost_evolution(cost_history, optimizer.config.get('grid', {}).get('description', 'qaoa_run'))
        
        analyze_results(counts)
        
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
