#!/usr/bin/env python3
"""
Benchmark script for QAOA wind turbine layout optimization
Fixed p=2, varying rhobeg values (0.3, 0.5, 0.7)
Generates grid visualization figures for article selection
"""

import subprocess
import json
import time
from datetime import datetime
import os

def run_qaoa_simulation(config_file, layers, rhobeg, repetition, grid_name):
    """Run a single QAOA simulation with specified parameters"""
    print(f"\n=== Running {grid_name} - p={layers}, rhobeg={rhobeg}, rep={repetition} ===")
    
    # Create modified config for this run
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Update QAOA parameters
    config['qaoa']['layers'] = layers
    config['qaoa']['optimizer_options']['rhobeg'] = rhobeg
    
    # Ensure uniform scores for consistency
    config['score']['mode'] = 'uniform'
    config['score']['uniform_value'] = 5.0
    
    # Increase shots for better statistics
    config['qaoa']['shots'] = 1024
    
    # Create temporary config file
    temp_config = f"temp_config_{grid_name}_p{layers}_rhobeg{rhobeg}_rep{repetition}.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        start_time = time.time()
        
        # Run QAOA simulation
        result = subprocess.run([
            'python', 'qaoa_turbinas.py', '-c', temp_config
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Success - Execution time: {execution_time:.2f}s")
            return True, execution_time
        else:
            print(f"✗ Error - Return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, execution_time
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout (300s)")
        return False, 300.0
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, 0.0
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config):
            os.remove(temp_config)

def main():
    """Execute systematic benchmarks with p=2 and varying rhobeg"""
    
    # Configuration for benchmarks
    configurations = [
        ("config.json", "2x3"),
        ("config_3x3.json", "3x3"), 
        ("config_4x4.json", "4x4")
    ]
    
    layers = 2  # Fixed p=2
    rhobeg_values = [0.3, 0.5, 0.7]
    repetitions = 3
    
    print(f"Starting systematic benchmark: p={layers}, rhobeg={rhobeg_values}")
    print(f"Repetitions per combination: {repetitions}")
    print(f"Total simulations: {len(configurations) * len(rhobeg_values) * repetitions}")
    
    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    total_time = 0.0
    
    start_benchmark = time.time()
    
    # Execute benchmarks for each configuration
    for config_file, grid_name in configurations:
        print(f"\n{'='*50}")
        print(f"GRID: {grid_name} ({config_file})")
        print(f"{'='*50}")
        
        if not os.path.exists(config_file):
            print(f"❌ Config file {config_file} not found!")
            continue
            
        # Test each rhobeg value
        for rhobeg in rhobeg_values:
            print(f"\n--- Testing rhobeg={rhobeg} ---")
            
            # Run multiple repetitions
            for rep in range(1, repetitions + 1):
                total_runs += 1
                success, exec_time = run_qaoa_simulation(
                    config_file, layers, rhobeg, rep, grid_name
                )
                
                total_time += exec_time
                
                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1
                
                # Brief pause between runs
                time.sleep(1)
    
    benchmark_time = time.time() - start_benchmark
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total simulations: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/total_runs*100:.1f}%")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Average time per simulation: {total_time/total_runs:.2f}s")
    print(f"Total benchmark time: {benchmark_time:.2f}s")
    print(f"Benchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_runs > 0:
        print(f"\n✓ Results saved to CSV files in resultados_simulacoes/")
        print(f"✓ Grid visualization figures saved to images/")
        print(f"✓ Ready for table generation and analysis")
    else:
        print(f"\n❌ No successful simulations completed")

if __name__ == "__main__":
    main()