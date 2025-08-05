#!/usr/bin/env python3
"""
Script para executar QAOA com diferentes configurações
"""

import sys
import os
import argparse
import json

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='QAOA para Otimização de Turbinas Eólicas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_config.py                           # Usa config.json padrão
  python run_config.py -c config_3x3.json       # Grid 3x3
  python run_config.py --config config_vertical.json  # Grid vertical
  python run_config.py --list-configs            # Lista configurações

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

def main():
    """Função principal"""
    args = parse_arguments()
    
    # Se solicitado, listar configurações e sair
    if args.list_configs:
        list_available_configs()
        return
    
    # Verificar se arquivo de configuração existe
    if not os.path.exists(args.config):
        print(f"❌ Arquivo de configuração não encontrado: {args.config}")
        print("\nArquivos disponíveis:")
        list_available_configs()
        sys.exit(1)
    
    print(f"🔧 Usando configuração: {args.config}")
    
    # Modificar temporariamente o arquivo de configuração padrão
    backup_file = None
    if args.config != 'config.json':
        # Fazer backup do config padrão se existir
        if os.path.exists('config.json'):
            backup_file = 'config.json.backup'
            os.rename('config.json', backup_file)
        
        # Copiar configuração escolhida para config.json
        import shutil
        shutil.copy(args.config, 'config.json')
    
    try:
        # Importar e executar o módulo principal
        import qaoa_turbinas
        print("\n" + "="*50)
        print("EXECUÇÃO CONCLUÍDA")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
    
    finally:
        # Restaurar backup se necessário
        if backup_file and args.config != 'config.json':
            os.remove('config.json')
            os.rename(backup_file, 'config.json')

if __name__ == "__main__":
    main()