#!/usr/bin/env python3
"""
Script para modificar configurações JSON para benchmarks
"""
import json
import sys

def modify_config(input_file, output_file, layers, rhobeg):
    """Modify config file with specific parameters"""
    with open(input_file, 'r') as f:
        config = json.load(f)
    
    # Update QAOA parameters
    config['qaoa']['layers'] = layers
    if 'optimizer_options' not in config['qaoa']:
        config['qaoa']['optimizer_options'] = {}
    config['qaoa']['optimizer_options']['rhobeg'] = rhobeg
    
    # Ensure uniform scores for consistency
    config['score']['mode'] = 'uniform'
    config['score']['uniform_value'] = 5.0
    
    # Increase shots for better statistics
    config['qaoa']['shots'] = 1024
    
    # Write modified config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python modify_config.py <input_config> <output_config> <layers> <rhobeg>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    layers = int(sys.argv[3])
    rhobeg = float(sys.argv[4])
    
    modify_config(input_file, output_file, layers, rhobeg)
    print(f"Config modificado: {input_file} -> {output_file} (p={layers}, rhobeg={rhobeg})")