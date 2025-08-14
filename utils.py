import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
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

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Gera gráficos (evolução do custo e trajetória γ-β) em images/'
    )

    parser.add_argument(
        '--plot-quantum',
        action='store_true',
        help='Plota representação do circuito quântico QAOA em images/'
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

def plot_cost_evolution(cost_history, config_name="config", save_plot=True, rhobeg=None):
    """Gera gráfico da evolução do custo durante a otimização QAOA"""
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(cost_history) + 1)
    plt.plot(iterations, cost_history, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    subtitle = f"Configuração: {config_name}"
    if rhobeg is not None:
        subtitle += f" | rhobeg={rhobeg}"
    plt.title(f'Evolução do Custo - QAOA Otimização\n{subtitle}', fontsize=14, fontweight='bold')
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
        base = config_name.replace('.json', '')
        if rhobeg is not None:
            base = f"{base}_rhobeg-{rhobeg}"
        filename = f"images/qaoa_optimization_{base}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico salvo como: {filename}")
        
        # Também salvar uma versão simples sem timestamp para fácil visualização
        base_simple = config_name.replace('.json', '')
        if rhobeg is not None:
            base_simple = f"{base_simple}_rhobeg-{rhobeg}"
        simple_filename = f"images/qaoa_latest_{base_simple}.png"
        plt.savefig(simple_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Versão simples salva como: {simple_filename}")
    
    # Não mostrar o plot na tela para evitar problemas em ambientes sem display
    # plt.show()
    plt.close()

def plot_gamma_beta_trajectory(gamma_history, beta_history, config_name="config", save_plot=True, rhobeg=None):
    """Plota a trajetória dos parâmetros γ (x) e β (y) ao longo das iterações.

    - gamma_history: lista de listas; gamma_history[l][t] é γ_l na iteração t
    - beta_history: lista de listas; beta_history[l][t] é β_l na iteração t
    """
    if not gamma_history or not beta_history:
        print("⚠️  Sem histórico de parâmetros para plotar.")
        return

    p = len(gamma_history)
    assert p == len(beta_history), "Tamanhos de histórico de γ e β não coincidem"

    plt.figure(figsize=(8, 6))

    # Cores distintas por caminhante
    colors = plt.cm.tab10.colors if p <= 10 else plt.cm.tab20.colors

    # Coletar todos os valores para determinar os limites dos eixos
    all_gammas = []
    all_betas = []

    for l in range(p):
        xs = gamma_history[l]
        ys = beta_history[l]
        if not xs or not ys:
            continue
        color = colors[l % len(colors)]
        plt.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.7, label=f'caminhante {l}')
        # Marcar início e fim
        plt.scatter(xs[0], ys[0], color=color, marker='o', s=60, edgecolors='k', linewidths=0.5)
        plt.scatter(xs[-1], ys[-1], color=color, marker='X', s=80, edgecolors='k', linewidths=0.5)

        # Pequenas setas para indicar direção (a cada ~N passos)
        if len(xs) > 1:
            skip = max(1, len(xs)//6)
            for i in range(0, len(xs)-1, skip):
                plt.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                             arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.6))
        
        # Coletar valores para limites dos eixos
        all_gammas.extend(xs)
        all_betas.extend(ys)

    # Ajustar limites dos eixos baseados no range dos dados
    if all_gammas and all_betas:
        import numpy as np
        
        gamma_min, gamma_max = min(all_gammas), max(all_gammas)
        beta_min, beta_max = min(all_betas), max(all_betas)
        
        # Ranges padrão QAOA: γ ∈ [0, 2π], β ∈ [0, π]
        gamma_default_min, gamma_default_max = 0, 2 * np.pi
        beta_default_min, beta_default_max = 0, np.pi
        
        # Expandir limites se os dados ultrapassarem o range padrão
        final_gamma_min = min(gamma_min, gamma_default_min)
        final_gamma_max = max(gamma_max, gamma_default_max)
        final_beta_min = min(beta_min, beta_default_min)
        final_beta_max = max(beta_max, beta_default_max)
        
        # Adicionar margem de 2% para visualização
        gamma_range = final_gamma_max - final_gamma_min
        beta_range = final_beta_max - final_beta_min
        margin = 0.02
        
        plt.xlim(final_gamma_min - margin * gamma_range, final_gamma_max + margin * gamma_range)
        plt.ylim(final_beta_min - margin * beta_range, final_beta_max + margin * beta_range)

    subtitle = f"Configuração: {config_name}"
    if rhobeg is not None:
        subtitle += f" | rhobeg={rhobeg}"
    plt.title(f'Trajetória dos Parâmetros (γ vs β)\n{subtitle}', fontsize=14, fontweight='bold')
    plt.xlabel('γ (gamma)', fontsize=12)
    plt.ylabel('β (beta)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Preparar área extra à direita para legendas externas
    ax = plt.gca()
    plt.subplots_adjust(right=0.75)

    # Legenda explicativa dos marcadores (início/fim)
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, linestyle='None'),
    ]

    # Legenda das camadas (p): fora do plot, canto superior direito externo
    leg_layers = ax.legend(title='Camadas (p)', loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax.add_artist(leg_layers)

    # Legenda dos marcadores: fora do plot, meio direito externo
    leg_markers = ax.legend(custom_lines, ['Início', 'Fim'], title='Marcadores', loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

    # Ajuste final
    plt.tight_layout()

    if save_plot:
        # Criar pasta images se não existir
        if not os.path.exists('images'):
            os.makedirs('images')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = config_name.replace('.json', '')
        if rhobeg is not None:
            base = f"{base}_rhobeg-{rhobeg}"
        filename = f"images/qaoa_params_traj_{base}_{timestamp}.png"
        # Incluir legendas externas no bounding box
        plt.savefig(filename, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg_layers, leg_markers))
        print(f"📊 Gráfico de trajetória salvo como: {filename}")

        base_simple = config_name.replace('.json', '')
        if rhobeg is not None:
            base_simple = f"{base_simple}_rhobeg-{rhobeg}"
        simple_filename = f"images/qaoa_params_traj_latest_{base_simple}.png"
        plt.savefig(simple_filename, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg_layers, leg_markers))
        print(f"📊 Versão simples salva como: {simple_filename}")

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

def load_ibm_config():
    """Carrega configurações da IBM do arquivo ibm.json"""
    ibm_file = "ibm.json"
    if not os.path.exists(ibm_file):
        raise FileNotFoundError(f"Arquivo {ibm_file} não encontrado. Crie o arquivo com suas configurações IBM.")
    
    with open(ibm_file, 'r') as f:
        ibm_config = json.load(f)
    
    return ibm_config

def confirm_ibm_execution(config):
    """Confirma execução no IBM Quantum com informações de custo"""
    try:
        ibm_config = load_ibm_config()
    except:
        print("❌ Arquivo ibm.json não encontrado. Usando configurações padrão.")
        ibm_config = {
            "instance": "meu_primeiro_computador_quantico",
            "backends": {"primary": "ibm_torino", "fallback": "ibm_brisbane"},
            "plan": {"type": "Open", "monthly_limit": "10 minutos"}
        }
    
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
    
    print(f"\n💰 Custo ({ibm_config['plan']['type']} Plan):")
    cost = ibm_config['plan'].get('cost_per_shot', 0.0) * total_shots
    if cost == 0:
        print(f"   • Custo: $0.00 (plano gratuito)")
    else:
        print(f"   • Custo estimado: ${cost:.2f}")
    
    print(f"\n🌐 Instância e Backends:")
    print(f"   • Instância: {ibm_config['instance']}")
    print(f"   • Backend primário: {ibm_config['backends']['primary']}")
    print(f"   • Backend fallback: {ibm_config['backends']['fallback']}")
    
    print(f"\n⏱️  Tempo estimado:")
    queue_info = ibm_config.get('queue_info', {})
    for backend, info in queue_info.items():
        print(f"   • {backend}: {info['qubits']} qubits, fila {info.get('typical_queue', 'N/A')}")
    print(f"   • Execução: 3-8min")
    print(f"   • Limite mensal: {ibm_config['plan']['monthly_limit']}")
    
    print(f"\n⚠️  IMPORTANTE:")
    print(f"   • Esta é uma execução em HARDWARE QUÂNTICO REAL")
    print(f"   • A execução pode falhar por problemas de hardware")
    print(f"   • Configurações definidas em ibm.json")
    
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

def create_grid_visualization(solution, optimizer, config_file, best_probability=None, total_score=None, wake_penalty=None):
    """Cria visualização gráfica do grid mais provável e salva na pasta images/"""
    
    # Criar pasta images se não existir
    if not os.path.exists('images'):
        os.makedirs('images')
        print("📁 Pasta images/ criada")
    
    # Configurar o grid como matriz
    grid_matrix = np.array(solution).reshape((optimizer.rows, optimizer.cols))
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(max(8, optimizer.cols * 1.5), max(6, optimizer.rows * 1.5)))
    
    # Criar colormap personalizado: 0 = branco (vazio), 1 = azul (turbina)
    colors = ['white', 'steelblue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plotar o grid
    im = ax.imshow(grid_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    
    # Adicionar grade
    ax.set_xticks(np.arange(-0.5, optimizer.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, optimizer.rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    
    # Configurar labels dos eixos
    ax.set_xticks(range(optimizer.cols))
    ax.set_yticks(range(optimizer.rows))
    ax.set_xticklabels([f'Col {i}' for i in range(optimizer.cols)])
    ax.set_yticklabels([f'Row {i}' for i in range(optimizer.rows)])
    
    # Adicionar números das posições e símbolos
    for i in range(optimizer.rows):
        for j in range(optimizer.cols):
            position = i * optimizer.cols + j
            if grid_matrix[i, j] == 1:
                # Turbina instalada
                ax.text(j, i, f'●\n{position}', ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='white')
            else:
                # Posição vazia (apenas número)
                ax.text(j, i, f'{position}', ha='center', va='center', 
                       fontsize=12, color='gray')
    
    # Mostrar direção do vento
    wind_direction = optimizer.wind_direction
    if wind_direction == (0, 1):
        wind_text = "Oeste → Leste"
        # Adicionar seta horizontal
        ax.annotate('', xy=(optimizer.cols-0.8, -0.3), xytext=(0.2, -0.3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.text(optimizer.cols/2, -0.5, 'VENTO', ha='center', va='top', 
               fontsize=10, color='red', fontweight='bold')
    elif wind_direction == (1, 0):
        wind_text = "Norte → Sul"
        # Adicionar seta vertical
        ax.annotate('', xy=(-0.3, optimizer.rows-0.2), xytext=(-0.3, 0.2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.text(-0.5, optimizer.rows/2, 'VENTO', ha='right', va='center', 
               fontsize=10, color='red', fontweight='bold', rotation=90)
    
    # Título com informações da simulação
    title = f'Grid de Turbinas Eólicas {optimizer.rows}x{optimizer.cols}\n'
    title += f'Direção do Vento: {wind_text}'
    
    if best_probability is not None:
        title += f' | Probabilidade: {best_probability:.1%}'
    if total_score is not None and wake_penalty is not None:
        net_score = total_score - wake_penalty
        title += f'\nScore Total: {total_score:.1f} - Penalidade: {wake_penalty:.1f} = Score Líquido: {net_score:.1f}'
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    # Remover ticks menores para limpeza visual
    ax.tick_params(which="minor", size=0)
    
    # Adicionar legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Turbina Instalada ●'),
        Patch(facecolor='white', edgecolor='gray', label='Posição Vazia ○')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Gerar nome do arquivo baseado na configuração
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = config_file.replace('.json', '').replace('config_', '').replace('config', 'default')
    num_turbines = sum(solution)
    
    filename = f'grid_visualization_{config_name}_{optimizer.rows}x{optimizer.cols}_{num_turbines}turbinas_{timestamp}.png'
    filepath = os.path.join('images', filename)
    
    # Salvar imagem
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # Fechar figura para liberar memória
    
    print(f"🖼️ Visualização do grid salva: {filepath}")
    return filepath

def plot_quantum_circuit(ansatz, config_file):
    """
    Plota o circuito quântico QAOA e salva na pasta images/
    
    Args:
        ansatz: Circuito quântico QAOA
        config_file: Nome do arquivo de configuração para identificar o plot
    """
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt
        
        # Garantir que a pasta images existe
        if not os.path.exists('images'):
            os.makedirs('images')
            print("📁 Pasta images/ criada")
        
        # Configurar o plot
        fig = plt.figure(figsize=(12, 8))
        
        # Desenhar o circuito usando o circuit_drawer do Qiskit
        # Usar output='mpl' para integração com matplotlib
        try:
            circuit_plot = circuit_drawer(
                ansatz, 
                output='mpl',
                style='iqp',  # Estilo IBM Quantum Platform
                plot_barriers=False,
                fold=-1,  # Não dobrar o circuito
                reverse_bits=False
            )
            
            # Adicionar título com informações
            n_qubits = ansatz.num_qubits
            n_params = len(ansatz.parameters)
            layers = len([p for p in ansatz.parameters if 'gamma' in str(p)])
            
            config_name = config_file.replace('.json', '').replace('config_', '').replace('config', 'default')
            title = f'Circuito Quântico QAOA - {config_name.upper()}\n'
            title += f'{n_qubits} qubits | {layers} camadas | {n_params} parâmetros'
            
            plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
            
            # Adicionar legenda explicativa
            legend_text = (
                "H: Hadamard (superposição inicial)\n"
                "RZ: Rotação Z (operador de custo γ)\n"
                "RZZ: Rotação ZZ (termos de interação)\n"
                "RX: Rotação X (operador de mistura β)"
            )
            
            plt.figtext(0.02, 0.02, legend_text, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
        except Exception as e:
            print(f"⚠️  Fallback: Usando representação de texto do circuito (erro no plot: {e})")
            # Fallback: criar uma visualização de texto simples
            fig, ax = plt.subplots(figsize=(12, 8))
            n_qubits = ansatz.num_qubits
            n_params = len(ansatz.parameters)
            layers = len([p for p in ansatz.parameters if 'gamma' in str(p)])
            
            ax.text(0.5, 0.5, str(ansatz), transform=ax.transAxes, 
                   fontfamily='monospace', fontsize=8, ha='center', va='center')
            ax.set_title(f'Circuito QAOA - {config_file}\n{n_qubits} qubits | {layers} camadas | {n_params} parâmetros', fontweight='bold')
            ax.axis('off')
        
        # Gerar nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_file.replace('.json', '').replace('config_', '').replace('config', 'default')
        filename = f'quantum_circuit_{config_name}_{n_qubits}qubits_{layers}layers_{timestamp}.png'
        filepath = os.path.join('images', filename)
        
        # Salvar o circuito
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"🎛️ Circuito quântico salvo: {filepath}")
        return filepath
        
    except ImportError as e:
        print(f"❌ Erro: Biblioteca necessária não encontrada para plotar circuito: {e}")
        print("   Instale com: pip install qiskit[visualization]")
        return None
    except Exception as e:
        print(f"❌ Erro ao plotar circuito quântico: {e}")
        return None
    
