import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
from datetime import datetime

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='QAOA para Otimização de Turbinas Eólicas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python qaoa_turbinas.py                           # Usa config.json padrão
  python qaoa_turbinas.py -c config_3x3.json       # Grid 3x3
  python qaoa_turbinas.py --config config_vertical.json  # Grid vertical
  python qaoa_turbinas.py --help                    # Mostra esta ajuda

Arquivos de configuração disponíveis:
  config.json         - Grid 2x3 padrão
  config_3x3.json     - Grid 3x3 com restrições
  config_vertical.json - Grid 4x2 com vento vertical
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Arquivo de configuração JSON (padrão: config.json)'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='Lista arquivos de configuração disponíveis'
    )
    
    parser.add_argument(
        '--benchmark-hamiltonian',
        action='store_true',
        help='Compara performance das implementações do Hamiltoniano'
    )
    
    parser.add_argument(
        '--ibm-quantum',
        action='store_true',
        help='Executa no computador quântico IBM (requer confirmação)'
    )
    
    return parser.parse_args()

def list_available_configs():
    """Lista arquivos de configuração disponíveis"""
    config_files = [f for f in os.listdir('.') if f.startswith('config') and f.endswith('.json')]
    
    if not config_files:
        print("❌ Nenhum arquivo de configuração encontrado!")
        return
    
    print("📁 Arquivos de configuração disponíveis:")
    for config_file in sorted(config_files):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                grid_info = config['grid']
                description = grid_info.get('description', 'Sem descrição')
                print(f"   {config_file:<20} - {description}")
        except Exception as e:
            print(f"   {config_file:<20} - ❌ Erro ao ler: {e}")

def load_config(config_file):
    """Carrega configurações do arquivo JSON"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

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

def validate_constraints(solution, optimizer):
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

def evaluate_solution(bitstring, score, wake_penalties, optimizer):
    """Avalia uma solução clássica considerando restrições"""
    x = [int(bit) for bit in bitstring]
    
    # Score total
    total_score = sum(x[i] * score[i] for i in range(optimizer.n_positions))
    
    # Penalidades de esteira
    wake_penalty = sum(x[i] * x[j] * penalty 
                      for (i, j), penalty in wake_penalties.items())
    
    # Penalidades de restrições
    constraint_penalty = 0
    if optimizer.enforce_constraints:
        num_turbines = sum(x)
        if num_turbines < optimizer.min_turbines:
            constraint_penalty += optimizer.constraint_penalty * (optimizer.min_turbines - num_turbines)
        if num_turbines > optimizer.max_turbines:
            constraint_penalty += optimizer.constraint_penalty * (num_turbines - optimizer.max_turbines)
    
    return total_score - wake_penalty - constraint_penalty

def show_active_penalties(solution, wake_penalties, positions_coords):
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

def bitstring_to_grid(bitstring):
    """Converte bitstring para representação de grid"""
    # Qiskit inverte a ordem dos bits
    solution = [int(bit) for bit in reversed(bitstring)]
    return solution

def display_interference_matrix(optimizer):
    """Exibe TODAS as combinações de turbinas, incluindo penalidades zero"""
    print(f"\n🌪️  MATRIZ COMPLETA DE INTERFERÊNCIAS (INCLUINDO ZEROS)")
    wind_desc = "Oeste → Leste" if optimizer.wind_direction == (0, 1) else "Norte → Sul"
    print(f"Direção do vento: {wind_desc}")
    print(f"Grid {optimizer.rows}x{optimizer.cols} - Analisando todas as {optimizer.n_positions * (optimizer.n_positions - 1)} combinações")
    print("="*70)
    
    # Calcular TODAS as combinações, não só as com penalidade > 0
    total_combinations = 0
    active_interferences = 0
    
    for i in range(optimizer.n_positions):
        source_coord = optimizer.positions_coords[i]
        print(f"\n📍 Turbina em {source_coord}:")
        
        targets_in_line = []
        targets_other = []
        
        for j in range(optimizer.n_positions):
            if i != j:
                target_coord = optimizer.positions_coords[j]
                penalty = optimizer.wake_penalties.get((i, j), 0.0)  # 0 se não existe
                
                # Calcular direção e distância
                if optimizer.wind_direction == (0, 1):  # oeste→leste
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

def plot_cost_evolution(cost_history, config_name="config", save_plot=True):
    """Gera gráfico da evolução do custo durante a otimização QAOA"""
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(cost_history) + 1)
    plt.plot(iterations, cost_history, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    plt.title(f'Evolução do Custo - QAOA Otimização\nConfiguração: {config_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Valor da Função de Custo', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Destacar o melhor valor
    min_cost = min(cost_history)
    min_iteration = cost_history.index(min_cost) + 1
    plt.plot(min_iteration, min_cost, 'ro', markersize=8, label=f'Melhor: {min_cost:.4f} (iter {min_iteration})')
    
    # Adicionar informações estatísticas
    plt.text(0.02, 0.98, f'Total de iterações: {len(cost_history)}\nMelhor custo: {min_cost:.4f}\nCusto final: {cost_history[-1]:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    if save_plot:
        # Criar pasta images se não existir
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # Gerar nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/qaoa_optimization_{config_name.replace('.json', '')}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico salvo como: {filename}")
        
        # Também salvar uma versão simples sem timestamp para fácil visualização
        simple_filename = f"images/qaoa_latest_{config_name.replace('.json', '')}.png"
        plt.savefig(simple_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Versão simples salva como: {simple_filename}")
    
    # Não mostrar o plot na tela para evitar problemas em ambientes sem display
    # plt.show()
    plt.close()

def load_ibm_api_key():
    """Carrega a API key da IBM do arquivo apikey.json"""
    api_file = "apikey.json"
    if not os.path.exists(api_file):
        raise FileNotFoundError(f"Arquivo {api_file} não encontrado. Crie o arquivo com sua API key da IBM.")
    
    with open(api_file, 'r') as f:
        api_data = json.load(f)
    
    api_key = api_data.get('apikey')
    if not api_key:
        raise ValueError(f"Campo 'apikey' não encontrado no arquivo {api_file}")
    
    return api_key

def confirm_ibm_execution(config):
    """Confirma execução no IBM Quantum com informações de custo"""
    print("\n" + "="*60)
    print("🚨 EXECUÇÃO NO IBM QUANTUM DETECTADA")
    print("="*60)
    
    # Calcular custos estimados
    shots = config.get("qaoa", {}).get("shots", 1024)
    max_iter = config.get("qaoa", {}).get("optimizer_options", {}).get("maxiter", 50)
    total_shots = shots * max_iter
    
    print(f"📊 Parâmetros configurados:")
    print(f"   • Shots por iteração: {shots}")
    print(f"   • Iterações máximas: {max_iter}")
    print(f"   • Total de shots: {total_shots:,}")
    
    print(f"\n💰 Custo (Plano Open - GRATUITO):")
    print(f"   • Custo: $0.00 (plano gratuito)")
    
    print(f"\n⏱️  Tempo estimado:")
    print(f"   • Fila de espera: ibm_brisbane (~1613 jobs), ibm_torino (~5379 jobs)")
    print(f"   • Execução: 3-8min")
    print(f"   • Tempo restante na instância: 10 minutos")
    
    print(f"\n⚠️  IMPORTANTE:")
    print(f"   • Esta é uma execução em HARDWARE QUÂNTICO REAL")
    print(f"   • O custo será cobrado da sua conta IBM")
    print(f"   • A execução pode falhar por problemas de hardware")
    
    while True:
        response = input(f"\n🤔 Deseja continuar? [y/N]: ").strip().lower()
        if response in ['y', 'yes', 'sim', 's']:
            print("✅ Execução confirmada. Iniciando...")
            return True
        elif response in ['n', 'no', 'não', 'nao', ''] or not response:
            print("❌ Execução cancelada.")
            return False
        else:
            print("❓ Resposta inválida. Digite 'y' para sim ou 'n' para não.")

def display_grid(solution, optimizer, title=None):
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
    
