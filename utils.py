import argparse
import os
import json

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
