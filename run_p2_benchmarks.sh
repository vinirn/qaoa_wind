#!/bin/bash

# Benchmark script for p=2 with varying rhobeg
echo "=== Benchmark QAOA p=2 com rhobeg variado ==="

# Clean old CSV files
echo "Limpando CSVs antigos..."
rm -f resultados_simulacoes/qaoa_resultados_*_oeste_leste_uniform_sem_restricoes.csv

# Configuration arrays
configs=("config.json" "config_3x3.json" "config_4x4.json")
grid_names=("2x3" "3x3" "4x4")
rhobeg_values=(0.3 0.5 0.7)
layers=2
repetitions=3

total_runs=$((${#configs[@]} * ${#rhobeg_values[@]} * repetitions))
current_run=0

echo "Total de execuções: $total_runs"
echo ""

# Execute benchmarks
for i in "${!configs[@]}"; do
    config_file="${configs[$i]}"
    grid_name="${grid_names[$i]}"
    
    echo "===========================================" 
    echo "GRID: $grid_name ($config_file)"
    echo "==========================================="
    
    for rhobeg in "${rhobeg_values[@]}"; do
        echo ""
        echo "--- Testando rhobeg=$rhobeg ---"
        
        for rep in $(seq 1 $repetitions); do
            current_run=$((current_run + 1))
            echo ""
            echo "=== Execução $current_run/$total_runs: $grid_name - p=$layers, rhobeg=$rhobeg, rep=$rep ==="
            
            # Create temporary config with modifications
            temp_config="temp_${grid_name}_p${layers}_rhobeg${rhobeg}_rep${rep}.json"
            
            # Copy base config and modify using Python script
            python modify_config.py "$config_file" "$temp_config" "$layers" "$rhobeg"
            
            # Run QAOA with plotting enabled
            echo "⏳ Executando com geração de figuras..."
            start_time=$(date +%s)
            
            # Modify run_qaoa.sh call to include --plot
            if timeout 300 bash -c "source qiskit_env/bin/activate && python qaoa_turbinas.py -c '$temp_config' --plot" > /dev/null 2>&1; then
                end_time=$(date +%s)
                execution_time=$((end_time - start_time))
                echo "✅ Sucesso - Tempo: ${execution_time}s"
            else
                echo "❌ Falha ou timeout"
            fi
            
            # Clean temp config
            rm -f "$temp_config"
            
            # Small delay
            sleep 1
        done
    done
done

echo ""
echo "=========================================="
echo "BENCHMARK CONCLUÍDO"
echo "=========================================="
echo "📊 Figuras de grid salvas em: images/"
echo "📈 Resultados salvos em: resultados_simulacoes/"
echo "✅ Pronto para análise"