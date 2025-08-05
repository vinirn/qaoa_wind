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
from utils import parse_arguments, list_available_configs, load_config
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import Estimator
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
        print(f"⚡ Penalidade máxima: {self.max_penalty}")
        
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
        self.setup_energy_production()
        self.wake_penalties = self.calculate_wake_penalties()
        
        
    def setup_grid(self):
        """Configura as coordenadas do grid"""
        self.positions_coords = {}
        for i in range(self.n_positions):
            row = i // self.cols
            col = i % self.cols
            self.positions_coords[i] = (row, col)
            
    def setup_energy_production(self):
        """Configura a produção de energia por posição"""
        energy_config = self.config["energy_production"]
        mode = energy_config["mode"]
        
        if mode == "fixed":
            self.energy = energy_config["values"][:self.n_positions]
            # Preencher com zeros se necessário
            while len(self.energy) < self.n_positions:
                self.energy.append(0.0)
        elif mode == "random":
            min_val, max_val = energy_config["random_range"]
            np.random.seed(42)  # Seed fixo para reprodutibilidade
            self.energy = np.random.uniform(min_val, max_val, self.n_positions).tolist()
        elif mode == "uniform":
            uniform_value = energy_config.get("uniform_value", 4.0)
            self.energy = [uniform_value] * self.n_positions
        else:
            raise ValueError(f"Modo de energia inválido: {mode}")
            
        print(f"⚡ Energia por posição: {[f'{e:.1f}' for e in self.energy]}")
        
    def display_interference_matrix(self):
        """Exibe TODAS as combinações de turbinas, incluindo penalidades zero"""
        print(f"\n🌪️  MATRIZ COMPLETA DE INTERFERÊNCIAS (INCLUINDO ZEROS)")
        wind_desc = "Oeste → Leste" if self.wind_direction == (0, 1) else "Norte → Sul"
        print(f"Direção do vento: {wind_desc}")
        print(f"Grid {self.rows}x{self.cols} - Analisando todas as {self.n_positions * (self.n_positions - 1)} combinações")
        print("="*70)
        
        # Calcular TODAS as combinações, não só as com penalidade > 0
        total_combinations = 0
        active_interferences = 0
        
        for i in range(self.n_positions):
            source_coord = self.positions_coords[i]
            print(f"\n📍 Turbina em {source_coord}:")
            
            targets_in_line = []
            targets_other = []
            
            for j in range(self.n_positions):
                if i != j:
                    target_coord = self.positions_coords[j]
                    penalty = self.wake_penalties.get((i, j), 0.0)  # 0 se não existe
                    
                    # Calcular direção e distância
                    if self.wind_direction == (0, 1):  # oeste→leste
                        dx = target_coord[1] - source_coord[1]
                        dy = target_coord[0] - source_coord[0]
                        same_line = (dy == 0)
                        in_wind_direction = (dx > 0)
                    else:  # norte→sul
                        dx = target_coord[0] - source_coord[0]
                        dy = target_coord[1] - source_coord[1]
                        same_line = (dy == 0)
                        in_wind_direction = (dx > 0)
                    
                    # Classificar as posições
                    status = ""
                    if penalty > 0:
                        status = f"💨 INTERFERE"
                        active_interferences += 1
                    elif in_wind_direction and same_line:
                        status = f"🔸 MESMA LINHA"
                    elif in_wind_direction:
                        status = f"➡️  VENTO"
                    else:
                        status = f"⚪ SEM EFEITO"
                    
                    info = f"   {target_coord}: {penalty:.2f} - {status}"
                    
                    if same_line and in_wind_direction:
                        targets_in_line.append(info)
                    else:
                        targets_other.append(info)
                    
                    total_combinations += 1
            
            # Mostrar primeiro as da mesma linha, depois outras
            for info in targets_in_line:
                print(info)
            for info in targets_other:
                print(info)
        
        print(f"\n📊 RESUMO FINAL:")
        print(f"   • Total de combinações analisadas: {total_combinations}")
        print(f"   • Interferências ativas (penalty > 0): {active_interferences}")
        print(f"   • Sem interferência (penalty = 0): {total_combinations - active_interferences}")
        print(f"   • Taxa de interferência: {active_interferences/total_combinations*100:.1f}%")
        print("="*70)
        
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



def get_config_file():
    """Determina qual arquivo de configuração usar"""
    if len(sys.argv) > 1:
        # Se há argumentos, processar normalmente
        args = parse_arguments()
        
        # Se solicitado, listar configurações e sair
        if args.list_configs:
            list_available_configs()
            sys.exit(0)
        
        # Verificar se arquivo de configuração existe
        if not os.path.exists(args.config):
            print(f"❌ Arquivo de configuração não encontrado: {args.config}")
            print("\nArquivos disponíveis:")
            list_available_configs()
            sys.exit(1)
        
        print(f"🔧 Usando configuração: {args.config}")
        return args.config
    else:
        # Sem argumentos, usar padrão
        return "config.json"

def run_optimization(optimizer):
    """Executa a otimização QAOA"""
    # Definir variáveis globais para compatibilidade
    global energy, positions_coords, wake_penalties
    energy = optimizer.energy
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
        counts, optimal_value = run_qaoa(p=1, max_iter=50)
        analyze_results(counts)
        
    except Exception as e:
        print(f"Erro na execução do QAOA: {e}")
        print("\nExecutando solução clássica de referência...")
        
        # Comparação clássica (apenas para grids pequenos)
        if optimizer.n_positions <= 20:  # Evitar explosão exponencial
            best_value = float('-inf')
            best_solution = None
            
            for i in range(2**optimizer.n_positions):
                bitstring = format(i, f'0{optimizer.n_positions}b')
                value = evaluate_solution(bitstring)
                if value > best_value:
                    best_value = value
                    best_solution = bitstring
            
            print(f"Melhor solução clássica: {best_solution}")
            print(f"Energia líquida: {best_value}")
        else:
            print("Grid muito grande para busca exaustiva clássica.")

# Instanciar o otimizador com arquivo apropriado
config_file = get_config_file()
optimizer = QAOATurbineOptimizer(config_file)

# Exibir matriz de interferências no início
optimizer.display_interference_matrix()

# Manter compatibilidade com código existente
energy = optimizer.energy
positions_coords = optimizer.positions_coords
wake_penalties = optimizer.wake_penalties

def create_cost_hamiltonian_ANTIGO():
    """Cria o Hamiltoniano de custo usando SparsePauliOp moderno"""
    pauli_list = []
    
    # Termos lineares (energia): -energy[i] * Z[i] 
    for i in range(optimizer.n_positions):
        pauli_list.append(("Z", [i], -energy[i]))  # Negativo para maximizar
    
    # Termos quadráticos (penalidades): penalty * Z[i] * Z[j]
    for (i, j), penalty in wake_penalties.items():
        pauli_list.append(("ZZ", [i, j], penalty))  # Positivo para penalizar
    
    # Usando from_sparse_list (método moderno recomendado)
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)
    
def create_cost_hamiltonian_QUADRATICO():
    """Cria o Hamiltoniano de custo com restrições min/max turbinas - VERSÃO QUADRÁTICA ORIGINAL"""
    pauli_list = []
    
    # Termos lineares (energia): -energy[i] * Z[i] 
    for i in range(optimizer.n_positions):
        pauli_list.append(("Z", [i], -energy[i]))  # Negativo para maximizar
    
    # Termos quadráticos (penalidades de esteira): penalty * Z[i] * Z[j]
    for (i, j), penalty in wake_penalties.items():
        pauli_list.append(("ZZ", [i, j], -penalty))  # CORREÇÃO: Negativo para penalizar no Hamiltoniano
    
    # NOVO: Restrições de número de turbinas
    if optimizer.enforce_constraints:
        # Para implementar restrições min/max, usamos penalidades quadráticas
        # que aproximam a função de contagem
        
        # Termo para penalizar muito poucas turbinas (< min_turbines)
        if optimizer.min_turbines > 0:
            # Adiciona penalidade crescente quando número de turbinas é baixo
            for i in range(optimizer.n_positions):
                for j in range(i+1, optimizer.n_positions):
                    # Penaliza quando AMBAS estão desligadas se estivermos abaixo do mínimo
                    # Implementação simplificada: penaliza ausência de pares
                    min_penalty = optimizer.constraint_penalty * (optimizer.min_turbines / optimizer.n_positions)
                    pauli_list.append(("II", [], min_penalty))  # Termo constante
                    pauli_list.append(("Z", [i], -min_penalty/2))
                    pauli_list.append(("Z", [j], -min_penalty/2))
                    pauli_list.append(("ZZ", [i, j], min_penalty/4))
        
        # Termo para penalizar muitas turbinas (> max_turbines)
        if optimizer.max_turbines < optimizer.n_positions:
            # Penaliza quando muitas turbinas estão ligadas
            max_penalty = optimizer.constraint_penalty
            for i in range(optimizer.n_positions):
                for j in range(i+1, optimizer.n_positions):
                    # Penaliza quando AMBAS estão ligadas se excedermos o máximo
                    excess_factor = max(0, (optimizer.n_positions - optimizer.max_turbines) / optimizer.n_positions)
                    pauli_list.append(("ZZ", [i, j], max_penalty * excess_factor))
    
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)

def create_cost_hamiltonian_LINEAR():
    """Cria o Hamiltoniano de custo com restrições min/max turbinas - VERSÃO LINEAR OTIMIZADA"""
    pauli_list = []
    
    # Termos lineares (energia): -energy[i] * Z[i] 
    for i in range(optimizer.n_positions):
        pauli_list.append(("Z", [i], -energy[i]))  # Negativo para maximizar
    
    # Termos quadráticos (penalidades de esteira): penalty * Z[i] * Z[j]
    for (i, j), penalty in wake_penalties.items():
        pauli_list.append(("ZZ", [i, j], -penalty))  # CORREÇÃO: Negativo para penalizar no Hamiltoniano
    
    # OTIMIZAÇÃO: Restrições com complexidade linear O(n) para construção
    if optimizer.enforce_constraints:
        # Implementação corrigida com penalizações mais fortes
        n = optimizer.n_positions
        penalty = optimizer.constraint_penalty
        
        # Estratégia híbrida: penalidades lineares E quadráticas mais eficazes
        if optimizer.min_turbines > 0 or optimizer.max_turbines < n:
            target_mid = (optimizer.min_turbines + optimizer.max_turbines) / 2.0
            
            # CORREÇÃO: Penalidades muito mais fortes para garantir cumprimento
            strong_penalty = penalty * 10  # Aumentar significativamente
            
            # Termo linear forte baseado no target
            # Se target_mid > n/2, incentivar turbinas ligadas (coef. negativo para Z)
            # Se target_mid < n/2, desincentivar turbinas ligadas (coef. positivo para Z)
            linear_coeff = strong_penalty * (1 - 2 * target_mid / n)
            
            for i in range(n):
                pauli_list.append(("Z", [i], linear_coeff))
            
            # Penalidade quadrática para reforçar a restrição
            # Implementa aproximadamente penalty * (sum(z_i) - target)²
            quad_penalty = strong_penalty / n
            
            # Termos ZZ que implementam (sum z_i)² de forma distribuída
            for i in range(n):
                for j in range(i+1, n):
                    # Coeficiente ajustado baseado no target
                    if target_mid > n / 2:
                        # Target alto: penalizar pouco as interações (permitir mais turbinas)
                        pauli_list.append(("ZZ", [i, j], -quad_penalty * 0.5))
                    else:
                        # Target baixo: penalizar muito as interações (forçar poucas turbinas)
                        pauli_list.append(("ZZ", [i, j], quad_penalty * 2.0))
    
    return SparsePauliOp.from_sparse_list(pauli_list, num_qubits=optimizer.n_positions)

def create_cost_hamiltonian():
    """FUNÇÃO PRINCIPAL - pode alternar entre implementações"""
    # Para debug: use a implementação quadrática que sabemos que funciona
    return create_cost_hamiltonian_QUADRATICO()
    # Para produção: quando linear estiver validada, usar:
    # return create_cost_hamiltonian_LINEAR()

def compare_hamiltonian_implementations():
    """Compara as duas implementações do Hamiltoniano para validação"""
    print("\n🔬 Comparando implementações do Hamiltoniano...")
    
    # Implementação quadrática (original)
    start_time = time.time()
    ham_quad = create_cost_hamiltonian_QUADRATICO()
    time_quad = time.time() - start_time
    
    # Implementação linear (otimizada)  
    start_time = time.time()
    ham_linear = create_cost_hamiltonian()
    time_linear = time.time() - start_time
    
    print(f"⏱️  Tempo construção quadrática: {time_quad:.4f}s")
    print(f"⏱️  Tempo construção linear: {time_linear:.4f}s")
    print(f"🚀 Speedup: {time_quad/time_linear:.2f}x")
    
    print(f"📊 Termos no Hamiltoniano quadrático: {len(ham_quad.paulis)}")
    print(f"📊 Termos no Hamiltoniano linear: {len(ham_linear.paulis)}")
    
    return ham_quad, ham_linear

# Importar time para benchmarking
import time
    
# 4. FUNÇÃO AUXILIAR PARA VALIDAR RESTRIÇÕES
def validate_constraints(solution):
    """Valida se uma solução atende às restrições"""
    num_turbines = sum(solution)
    
    violations = []
    if optimizer.enforce_constraints:
        if num_turbines < optimizer.min_turbines:
            deficit = optimizer.min_turbines - num_turbines
            violations.append(f"❌ MUITO POUCAS turbinas: {num_turbines} < {optimizer.min_turbines} (faltam {deficit})")
        if num_turbines > optimizer.max_turbines:
            excess = num_turbines - optimizer.max_turbines
            violations.append(f"❌ MUITAS turbinas: {num_turbines} > {optimizer.max_turbines} (excesso de {excess})")
        if len(violations) == 0:
            violations.append(f"✅ Restrições ATENDIDAS: {num_turbines} turbinas está no intervalo [{optimizer.min_turbines}, {optimizer.max_turbines}]")
    else:
        violations.append(f"⚠️  Sem restrições ativas: {num_turbines} turbinas (qualquer número permitido)")
    
    return violations

def create_qaoa_ansatz(cost_hamiltonian, p=1):
    """Cria o ansatz QAOA usando QAOAAnsatz moderno"""
    # Usando a classe QAOAAnsatz (método recomendado)
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
    return ansatz

  
def evaluate_solution(bitstring):
    """Avalia uma solução clássica considerando restrições"""
    x = [int(bit) for bit in bitstring]
    
    # Energia total
    total_energy = sum(x[i] * energy[i] for i in range(optimizer.n_positions))
    
    # Penalidades de esteira
    wake_penalty = sum(x[i] * x[j] * penalty 
                      for (i, j), penalty in wake_penalties.items())
    
    # NOVO: Penalidades de restrições
    constraint_penalty = 0
    if optimizer.enforce_constraints:
        num_turbines = sum(x)
        if num_turbines < optimizer.min_turbines:
            constraint_penalty += optimizer.constraint_penalty * (optimizer.min_turbines - num_turbines)
        if num_turbines > optimizer.max_turbines:
            constraint_penalty += optimizer.constraint_penalty * (num_turbines - optimizer.max_turbines)
    
    return total_energy - wake_penalty - constraint_penalty


def show_active_penalties(solution):
    """Mostra apenas as penalidades ativas para a solução atual"""
    active_penalties = []
    total_penalty = 0
    
    for (i, j), penalty in wake_penalties.items():
        if solution[i] == 1 and solution[j] == 1:  # Ambas turbinas instaladas
            coord1 = positions_coords[i]
            coord2 = positions_coords[j]
            active_penalties.append((coord1, coord2, penalty))
            total_penalty += penalty
    
    if active_penalties:
        print(f"\n🌪️  INTERFERÊNCIAS ATIVAS:")
        for coord1, coord2, penalty in active_penalties:
            print(f"   {coord1} → {coord2}: penalidade {penalty}")
        print(f"   Total de penalidades: {total_penalty}")
    else:
        print(f"\n✅ NENHUMA INTERFERÊNCIA ATIVA (configuração otimizada!)")
    
    return total_penalty

def run_qaoa(p=1, max_iter=50):
    """Executa o algoritmo QAOA usando APIs modernas"""
    print(f"\nConfigurando QAOA com p={p} camadas...")
    
    # Criar Hamiltoniano e ansatz
    cost_hamiltonian = create_cost_hamiltonian()
    ansatz = create_qaoa_ansatz(cost_hamiltonian, p)
    
    print(f"Ansatz criado com {ansatz.num_qubits} qubits e {len(ansatz.parameters)} parâmetros")
    
    # Estimator para calcular valores esperados
    estimator = Estimator()
    
    def cost_function(params):
        """Função de custo para otimização clássica"""
        # Executar com primitives
        job = estimator.run([ansatz], [cost_hamiltonian], [params])
        result = job.result()
        return result.values[0]
    
    # Otimização clássica dos parâmetros
    print("Iniciando otimização dos parâmetros...")
    
    # Chute inicial: valores pequenos aleatórios
    initial_params = np.random.uniform(0, 0.1, len(ansatz.parameters))
    
    # Otimizar usando COBYLA (robusto para otimização ruidosa)
    result = minimize(cost_function, initial_params, method='COBYLA', 
                     options={'maxiter': max_iter})
    
    print(f"Otimização concluída em {result.nfev} avaliações")
    print(f"Valor ótimo encontrado: {result.fun}")
    
    # Executar circuito final para obter distribuição
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    
    # Preparar circuito final com medições
    final_circuit = ansatz.assign_parameters(result.x)
    final_circuit.measure_all()
    
    # Simular
    simulator = AerSimulator()
    transpiled_circuit = transpile(final_circuit, simulator)
    job = simulator.run(transpiled_circuit, shots=1024)
    counts = job.result().get_counts()
    
    return counts, result.fun

def bitstring_to_grid(bitstring):
    """Converte bitstring para representação de grid 2x3"""
    # Qiskit inverte a ordem dos bits
    solution = [int(bit) for bit in reversed(bitstring)]
    return solution

def display_grid(solution, title=None):
    """Exibe o grid de forma visual dinamicamente"""
    if title is None:
        title = f"Grid {optimizer.rows}x{optimizer.cols}"
        
    print(f"\n{title}:")
    
    # Cabeçalho das colunas
    header = "    " + "".join(f"Col {c:2d} " for c in range(optimizer.cols))
    print(header)
    
    # Linha superior da tabela
    line_top = "   ┌" + "─────┬" * (optimizer.cols - 1) + "─────┐"
    print(line_top)
    
    # Linhas do grid
    for row in range(optimizer.rows):
        # Valores da linha
        values = []
        for col in range(optimizer.cols):
            i = row * optimizer.cols + col
            values.append(f"  {solution[i]}  ")
        
        row_line = f"L{row} │" + "│".join(values) + "│"
        print(row_line)
        
        # Linha separadora (exceto na última linha)
        if row < optimizer.rows - 1:
            line_mid = "   ├" + "─────┼" * (optimizer.cols - 1) + "─────┤"
            print(line_mid)
    
    # Linha inferior da tabela
    line_bottom = "   └" + "─────┴" * (optimizer.cols - 1) + "─────┘"
    print(line_bottom)
    
    # Mostrar coordenadas das turbinas instaladas
    installed = []
    for i in range(optimizer.n_positions):
        if solution[i] == 1:
            row = i // optimizer.cols
            col = i % optimizer.cols
            installed.append(f"({row},{col})")
    
    if installed:
        print(f"Turbinas em: {', '.join(installed)}")
    else:
        print("Nenhuma turbina instalada")

def analyze_results_ANTIGO(counts):
    """Analisa os resultados do QAOA"""
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
    display_grid(solution, "MELHOR CONFIGURAÇÃO")
    
    # Mostrar penalidades efetivas
    total_penalty = show_active_penalties(solution)
    
    # Métricas
    installed_positions = [i for i in range(optimizer.n_positions) if solution[i] == 1]
    total_energy = sum(solution[i] * energy[i] for i in range(optimizer.n_positions))
    
    print(f"\n📊 MÉTRICAS:")
    print(f"   • Número de turbinas: {len(installed_positions)}")
    print(f"   • Produção total: {total_energy}")
    print(f"   • Penalidades: {total_penalty}")
    print(f"   • Energia líquida: {total_energy - total_penalty}")
    
    # Mostrar produção por posição
    print(f"\n⚡ PRODUÇÃO POR POSIÇÃO:")
    for i in range(optimizer.n_positions):
        row, col = i // optimizer.cols, i % optimizer.cols
        status = "🟢 ATIVA" if solution[i] == 1 else "⚪ VAZIA"
        prod = energy[i] if solution[i] == 1 else 0
        print(f"   ({row},{col}) - {status} - Energia: {prod:.1f}")
    
    # Top 5 soluções com visualização
    print(f"\n🏆 TOP 5 SOLUÇÕES MAIS PROVÁVEIS:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for i, (bitstring, count) in enumerate(sorted_counts[:5]):
        prob = count / sum(counts.values())
        value = evaluate_solution(bitstring)
        sol = bitstring_to_grid(bitstring)
        
        # Formato compacto do grid
        if optimizer.rows == 2 and optimizer.cols == 3:
            grid_compact = f"[{sol[0]}{sol[1]}{sol[2]}|{sol[3]}{sol[4]}{sol[5]}]"
        else:
            # Para grids maiores, mostrar apenas as posições ativas
            active_pos = [str(i) for i in range(optimizer.n_positions) if sol[i] == 1]
            grid_compact = f"[{','.join(active_pos) if active_pos else 'vazio'}]"
        
        print(f"   {i+1}. {grid_compact} - {prob:.3f} - Energia: {value}")
        
        # Mostrar grid completo apenas para as 3 melhores
        if i < 3:
            display_grid(sol, f"Solução #{i+1}")
            

# 6. MODIFICAÇÃO NA FUNÇÃO analyze_results
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
    display_grid(solution, "MELHOR CONFIGURAÇÃO")
    
    # NOVO: Validar restrições de forma destacada
    violations = validate_constraints(solution)
    print(f"\n🎯 VERIFICAÇÃO DE RESTRIÇÕES:")
    for violation in violations:
        print(f"   {violation}")
    
    # Mostrar penalidades efetivas
    total_penalty = show_active_penalties(solution)
    
    # Métricas detalhadas
    installed_positions = [i for i in range(optimizer.n_positions) if solution[i] == 1]
    total_energy = sum(solution[i] * energy[i] for i in range(optimizer.n_positions))
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
    
    print(f"   • Produção total: {total_energy:.2f}")
    print(f"   • Penalidades de esteira: {total_penalty:.2f}")
    net_energy = total_energy - total_penalty
    if optimizer.enforce_constraints and constraint_penalty > 0:
        net_energy -= constraint_penalty
        print(f"   • Energia líquida final: {net_energy:.2f} (descontando restrições)")
    else:
        print(f"   • Energia líquida: {net_energy:.2f}")
    
    # Mostrar produção por posição
    print(f"\n⚡ PRODUÇÃO POR POSIÇÃO:")
    for i in range(optimizer.n_positions):
        row = i // optimizer.cols
        col = i % optimizer.cols
        status = "🌪️" if solution[i] == 1 else "⬜"
        prod = energy[i] if solution[i] == 1 else 0
        print(f"   Pos ({row},{col}): {status} Produção: {prod:.1f}")
        
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
    print(f"🏆 Energia líquida final: {net_energy:.2f}")


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
    energy = optimizer.energy
    wake_penalties = optimizer.wake_penalties
    
    # Se solicitado benchmark, executar e sair
    if args.benchmark_hamiltonian:
        compare_hamiltonian_implementations()
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
        max_iterations = optimizer.config["qaoa"]["max_iterations"]
        print(f"\n🔧 QAOA: {p_layers} camadas, {max_iterations} iterações")
        
        # Executar QAOA
        counts, optimal_value = run_qaoa(p=p_layers, max_iter=max_iterations)
        analyze_results(counts)
        
    except Exception as e:
        print(f"Erro na execução do QAOA: {e}")
        print("\nExecutando solução clássica de referência...")
        
        # Comparação clássica (apenas para grids pequenos)
        if optimizer.n_positions <= 20:  # Evitar explosão exponencial
            best_value = float('-inf')
            best_solution = None
            
            for i in range(2**optimizer.n_positions):
                bitstring = format(i, f'0{optimizer.n_positions}b')
                value = evaluate_solution(bitstring)
                if value > best_value:
                    best_value = value
                    best_solution = bitstring
            
            print(f"Melhor solução clássica: {best_solution}")
            print(f"Energia líquida: {best_value}")
        else:
            print("Grid muito grande para busca exaustiva clássica.")
