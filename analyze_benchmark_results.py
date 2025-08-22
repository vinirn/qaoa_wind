#!/usr/bin/env python3
"""
Analyze QAOA benchmark results and generate LaTeX tables
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """Load all CSV result files"""
    results = {}
    
    # Load 2x3 results
    df_2x3 = pd.read_csv('resultados_simulacoes/qaoa_resultados_2x3_oeste_leste_uniform_sem_restricoes.csv')
    # Remove rows where essential columns are NaN, but keep rows with only constraint columns NaN
    df_2x3 = df_2x3.dropna(subset=['net_score', 'layers', 'rhobeg', 'best_probability'])
    results['2x3'] = df_2x3
    
    # Load 3x3 results  
    df_3x3 = pd.read_csv('resultados_simulacoes/qaoa_resultados_3x3_oeste_leste_uniform_sem_restricoes.csv')
    df_3x3 = df_3x3.dropna(subset=['net_score', 'layers', 'rhobeg', 'best_probability'])
    results['3x3'] = df_3x3
    
    # Load 4x4 results
    df_4x4 = pd.read_csv('resultados_simulacoes/qaoa_resultados_4x4_oeste_leste_uniform_sem_restricoes.csv') 
    df_4x4 = df_4x4.dropna(subset=['net_score', 'layers', 'rhobeg', 'best_probability'])
    results['4x4'] = df_4x4
    
    return results

def analyze_by_grid_layers_rhobeg(results):
    """Analyze results grouped by grid size, layers, and rhobeg"""
    analysis = {}
    
    for grid_size, df in results.items():
        analysis[grid_size] = {}
        
        # Group by layers and rhobeg
        grouped = df.groupby(['layers', 'rhobeg'])
        
        for (layers, rhobeg), group in grouped:
            key = f"p{layers}_r{rhobeg}"
            
            stats = {
                'layers': layers,
                'rhobeg': rhobeg,
                'runs': len(group),
                'avg_net_score': group['net_score'].mean(),
                'max_net_score': group['net_score'].max(), 
                'min_net_score': group['net_score'].min(),
                'std_net_score': group['net_score'].std(),
                'avg_prob': group['best_probability'].mean() * 100,
                'max_prob': group['best_probability'].max() * 100,
                'avg_time': group['execution_time_s'].mean(),
                'avg_turbines': group['num_turbines'].mean(),
                'avg_penalty_ratio': group['penalty_ratio'].mean() * 100,
                'best_config': group.loc[group['net_score'].idxmax(), 'best_bitstring'],
                'convergence_rate': (group['iterations'] < group['maxiter']).mean() * 100
            }
            
            analysis[grid_size][key] = stats
            
    return analysis

def generate_detailed_table(analysis):
    """Generate detailed LaTeX table for each grid size"""
    tables = {}
    
    for grid_size in ['2x3', '3x3', '4x4']:
        if grid_size not in analysis:
            continue
            
        table_data = []
        data = analysis[grid_size]
        
        # Sort by layers then rhobeg
        sorted_keys = sorted(data.keys(), key=lambda x: (data[x]['layers'], data[x]['rhobeg']))
        
        for key in sorted_keys:
            stats = data[key]
            row = [
                str(stats['layers']),
                str(stats['rhobeg']), 
                f"{stats['avg_prob']:.1f}",
                f"{stats['max_prob']:.1f}",
                f"{stats['avg_net_score']:.1f}",
                f"{stats['max_net_score']:.1f}",
                f"{stats['avg_turbines']:.1f}",
                f"{stats['avg_time']:.2f}",
                f"{stats['convergence_rate']:.0f}"
            ]
            table_data.append(row)
        
        # Generate LaTeX table
        qubits = {'2x3': 6, '3x3': 9, '4x4': 16}[grid_size]
        
        latex = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Resultados QAOA para Grid {grid_size} ({qubits} qubits) - Análise por Camadas e rhobeg}}
\\label{{tab:resultados_{grid_size.replace('x', '_')}}}
\\footnotesize
\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|}}
\\hline
\\textbf{{Camadas}} & \\textbf{{rhobeg}} & \\textbf{{Prob. Média (\\%)}} & \\textbf{{Prob. Máx (\\%)}} & \\textbf{{Score Médio}} & \\textbf{{Score Máximo}} & \\textbf{{Turbinas Médias}} & \\textbf{{Tempo (s)}} & \\textbf{{Converg. (\\%)}} \\\\
\\hline"""
        
        for row in table_data:
            latex += "\n" + " & ".join(row) + " \\\\"
        
        latex += """
\\hline
\\end{tabular}
\\end{table*}"""
        
        tables[grid_size] = latex
        
    return tables

def generate_summary_table(analysis):
    """Generate summary comparison table across all grids"""
    
    # Find best results for each grid
    best_results = {}
    
    for grid_size, data in analysis.items():
        best_score = -float('inf')
        best_config = None
        
        for config, stats in data.items():
            if stats['max_net_score'] > best_score:
                best_score = stats['max_net_score']
                best_config = stats
        
        if best_config:
            best_results[grid_size] = {
                'qubits': {'2x3': 6, '3x3': 9, '4x4': 16}[grid_size],
                'best_score': best_score,
                'best_layers': best_config['layers'],
                'best_rhobeg': best_config['rhobeg'],
                'best_prob': best_config['max_prob'],
                'avg_time': best_config['avg_time'],
                'avg_turbines': best_config['avg_turbines']
            }
    
    # Generate LaTeX table
    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Comparação de Performance QAOA: Melhores Resultados por Grid}
\\label{tab:benchmark_comparison}
\\footnotesize
\\begin{tabular}{|c|c|c|c|c|c|c|c|}
\\hline
\\textbf{Grid} & \\textbf{Qubits} & \\textbf{Score Máximo} & \\textbf{Camadas Ótimas} & \\textbf{rhobeg Ótimo} & \\textbf{Prob. Máx (\\%)} & \\textbf{Turbinas Médias} & \\textbf{Tempo Médio (s)} \\\\
\\hline"""
    
    for grid_size in ['2x3', '3x3', '4x4']:
        if grid_size in best_results:
            res = best_results[grid_size]
            latex += f"""
{grid_size} & {res['qubits']} & {res['best_score']:.1f} & {res['best_layers']} & {res['best_rhobeg']} & {res['best_prob']:.1f} & {res['avg_turbines']:.1f} & {res['avg_time']:.2f} \\\\"""
    
    latex += """
\\hline
\\multicolumn{8}{|l|}{\\textbf{Observações dos Novos Benchmarks:}} \\\\
\\multicolumn{8}{|l|}{• Testados diferentes valores de rhobeg (0.3, 0.5, 0.7) sistematicamente} \\\\
\\multicolumn{8}{|l|}{• 3 repetições por combinação para maior confiabilidade estatística} \\\\
\\multicolumn{8}{|l|}{• Performance otimizada com configurações específicas por grid} \\\\
\\multicolumn{8}{|l|}{• Maior variabilidade observada em função dos parâmetros do otimizador} \\\\
\\end{tabular}
\\end{table*}"""
    
    return latex

def main():
    print("🔬 Analyzing QAOA benchmark results...")
    
    # Load results
    results = load_results()
    
    print(f"Loaded results:")
    for grid, df in results.items():
        print(f"  {grid}: {len(df)} runs")
    
    # Analyze by grid, layers, and rhobeg
    analysis = analyze_by_grid_layers_rhobeg(results)
    
    # Generate detailed tables
    detailed_tables = generate_detailed_table(analysis)
    
    # Generate summary table
    summary_table = generate_summary_table(analysis)
    
    # Save tables to files
    output_dir = Path("tabelas_atualizadas")
    output_dir.mkdir(exist_ok=True)
    
    for grid_size, table in detailed_tables.items():
        with open(output_dir / f"tabela_{grid_size.replace('x', '_')}_detalhada.tex", 'w') as f:
            f.write(table)
        print(f"✅ Generated detailed table for {grid_size}")
    
    with open(output_dir / "tabela_comparacao_geral.tex", 'w') as f:
        f.write(summary_table)
    print(f"✅ Generated summary comparison table")
    
    # Print some statistics
    print("\n📊 Key Statistics:")
    for grid_size, data in analysis.items():
        best_score = max(stats['max_net_score'] for stats in data.values())
        best_config = max(data.values(), key=lambda x: x['max_net_score'])
        print(f"{grid_size}: Best score = {best_score:.1f} (p={best_config['layers']}, rhobeg={best_config['rhobeg']})")
    
    print(f"\n✅ Analysis complete! Check tabelas_atualizadas/ for LaTeX tables")

if __name__ == "__main__":
    main()