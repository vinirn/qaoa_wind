#!/usr/bin/env python3
"""
Systematic benchmark runner for QAOA wind turbine optimization
"""
import json
import subprocess
import sys
import time
from pathlib import Path

def update_config(config_file, layers, rhobeg):
    """Update configuration file with new parameters"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    config['qaoa']['layers'] = layers
    config['qaoa']['optimizer_options']['rhobeg'] = rhobeg
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated config: layers={layers}, rhobeg={rhobeg}")

def run_qaoa(config_file):
    """Run QAOA with given configuration"""
    cmd = ['./run_qaoa.sh', config_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ Run completed successfully")
            return True
        else:
            print(f"✗ Run failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Run timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Run error: {e}")
        return False

def main():
    # Configuration parameters
    grids = {
        '2x3': 'config_2x3_benchmark.json',
        '3x3': 'config_3x3.json', 
        '4x4': 'config_4x4.json'
    }
    
    layers_values = [1, 2, 3]
    rhobeg_values = [0.3, 0.5, 0.7]
    repetitions = 3
    
    total_runs = len(grids) * len(layers_values) * len(rhobeg_values) * repetitions
    current_run = 0
    
    print(f"Starting systematic benchmark: {total_runs} total runs")
    print("=" * 60)
    
    for grid_name, config_file in grids.items():
        print(f"\n🔬 GRID {grid_name.upper()} BENCHMARKS")
        print("-" * 40)
        
        for layers in layers_values:
            for rhobeg in rhobeg_values:
                print(f"\n📊 Config: p={layers}, rhobeg={rhobeg}")
                
                for rep in range(repetitions):
                    current_run += 1
                    print(f"[{current_run:2d}/{total_runs}] Rep {rep+1}/3... ", end="", flush=True)
                    
                    # Update configuration
                    update_config(config_file, layers, rhobeg)
                    
                    # Run QAOA
                    success = run_qaoa(config_file)
                    
                    if not success:
                        print(f"⚠️  Failed run: {grid_name}, p={layers}, rhobeg={rhobeg}, rep={rep+1}")
                    
                    # Brief pause between runs
                    time.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"✅ Benchmark completed! Total runs: {current_run}")
    print("Check the CSV files in resultados_simulacoes/ for results")

if __name__ == "__main__":
    main()