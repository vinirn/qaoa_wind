#!/usr/bin/env python3
"""
Script para analisar correlações dos resultados QAOA.

Uso:
    python plot_net_vs_final.py <arquivo_csv>
    
Exemplo:
    python plot_net_vs_final.py qaoa_resultados_4x4_oeste_leste_uniform_sem_restricoes.csv
    
O script gera dois scatter plots mostrando as correlações entre:

SUBPLOT 1 - Validação do Hamiltoniano:
- net_score: Score real da configuração (score - penalty)  
- final_cost: Valor que o QAOA minimiza (valor de expectativa do Hamiltoniano)
- Esperado: final_cost ≈ -net_score (correlação negativa forte)

SUBPLOT 2 - Qualidade da Otimização:
- net_score: Score real da configuração (score - penalty)
- best_probability: Probabilidade de medir a melhor solução
- Esperado: Correlação positiva (QAOA deveria favorecer soluções melhores)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_correlation(csv_file):
    """
    Plota correlação entre net_score e final_cost.
    
    Args:
        csv_file (str): Caminho para o arquivo CSV com resultados
    """
    
    # Verificar se arquivo existe
    if not os.path.exists(csv_file):
        print(f"❌ Arquivo não encontrado: {csv_file}")
        return
    
    try:
        # Carregar dados
        print(f"📊 Carregando dados de: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Verificar colunas necessárias
        required_cols = ['net_score', 'final_cost', 'best_probability']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Colunas faltando: {missing_cols}")
            print(f"📋 Colunas disponíveis: {list(df.columns)}")
            return
            
        # Extrair dados
        net_score = df['net_score'].values
        final_cost = df['final_cost'].values
        best_probability = df['best_probability'].values
        
        print(f"📈 Total de pontos: {len(net_score)}")
        print(f"📊 Range net_score: [{net_score.min():.1f}, {net_score.max():.1f}]")
        print(f"📊 Range final_cost: [{final_cost.min():.1f}, {final_cost.max():.1f}]")
        print(f"📊 Range best_probability: [{best_probability.min():.4f}, {best_probability.max():.4f}]")
        
        # Calcular correlações
        correlation_cost = np.corrcoef(net_score, final_cost)[0, 1]
        correlation_prob = np.corrcoef(net_score, best_probability)[0, 1]
        
        # Calcular correlação esperada (final_cost = -net_score)
        expected_final_cost = -net_score
        expected_correlation = np.corrcoef(final_cost, expected_final_cost)[0, 1]
        
        print(f"🔗 Correlação net_score vs final_cost: {correlation_cost:.3f}")
        print(f"🔗 Correlação net_score vs best_probability: {correlation_prob:.3f}")
        print(f"🎯 Correlação esperada (final_cost ≈ -net_score): {expected_correlation:.3f}")
        
        # Criar figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # SUBPLOT 1: Net Score vs Final Cost
        ax1.scatter(net_score, final_cost, alpha=0.7, s=60, c='blue')
        ax1.set_xlabel('Net Score (score - penalty)', fontsize=12)
        ax1.set_ylabel('Final Cost (QAOA minimize)', fontsize=12)
        ax1.set_title('Net Score vs Final Cost', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Adicionar texto com estatísticas do subplot 1
        stats_text1 = f'Correlação: {correlation_cost:.3f}\n'
        stats_text1 += f'Esperada: {expected_correlation:.3f}\n'
        stats_text1 += f'Pontos: {len(net_score)}'
        
        ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # SUBPLOT 2: Net Score vs Best Probability  
        ax2.scatter(net_score, best_probability, alpha=0.7, s=60, c='green')
        ax2.set_xlabel('Net Score (score - penalty)', fontsize=12)
        ax2.set_ylabel('Best Probability', fontsize=12)
        ax2.set_title('Net Score vs Best Probability', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Adicionar texto com estatísticas do subplot 2
        stats_text2 = f'Correlação: {correlation_prob:.3f}\n'
        stats_text2 += f'Esperada: positiva\n'
        stats_text2 += f'Pontos: {len(net_score)}'
        
        ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Título geral
        fig.suptitle(f'Análise de Correlações QAOA\n{os.path.basename(csv_file)}', fontsize=16)
        
        # Salvar plot
        output_file = csv_file.replace('.csv', '_correlation_plot.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Plot salvo em: {output_file}")
        
        # Mostrar plot
        plt.show()
        
        # Análise dos resultados
        print("\n" + "="*50)
        print("📋 ANÁLISE DOS RESULTADOS:")
        print("="*50)
        
        # Análise do final_cost vs net_score
        print("\n🔍 ANÁLISE FINAL_COST vs NET_SCORE:")
        if abs(correlation_cost + 1) < 0.1:  # Correlação próxima de -1
            print("✅ HAMILTONIANO CORRETO: Correlação forte negativa detectada")
        elif abs(correlation_cost) < 0.3:  # Correlação fraca
            print("⚠️  PROBLEMA: Correlação muito fraca - Hamiltoniano pode estar incorreto")
        elif correlation_cost > 0.5:  # Correlação positiva forte
            print("❌ PROBLEMA CRÍTICO: Correlação positiva - Sinais invertidos no Hamiltoniano")
        else:
            print("🔍 INVESTIGAR: Correlação moderada - Verificar implementação")
            
        if abs(expected_correlation + 1) < 0.1:  # Correlação esperada próxima de -1
            print("✅ DADOS CONSISTENTES: Correlação com linha ideal próxima de -1")
        else:
            print(f"⚠️  INCONSISTÊNCIA: Correlação com ideal = {expected_correlation:.2f} (esperado ≈ -1)")
        
        # Análise do best_probability vs net_score  
        print(f"\n🎯 ANÁLISE BEST_PROBABILITY vs NET_SCORE:")
        if correlation_prob > 0.5:  # Correlação positiva forte
            print("✅ COMPORTAMENTO ESPERADO: Soluções melhores têm maior probabilidade")
        elif abs(correlation_prob) < 0.3:  # Correlação fraca
            print("⚠️  PROBLEMA: QAOA não está convergindo para soluções ótimas")
        elif correlation_prob < -0.3:  # Correlação negativa
            print("❌ PROBLEMA CRÍTICO: QAOA favorece soluções piores - Verificar implementação")
        else:
            print("🔍 INVESTIGAR: Correlação moderada - Pode precisar mais iterações ou melhor configuração")
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")

def main():
    if len(sys.argv) != 2:
        print("❌ Uso: python plot_net_vs_final.py <arquivo_csv>")
        print("\n📝 Exemplo:")
        print("   python plot_net_vs_final.py qaoa_resultados_4x4_oeste_leste_uniform_sem_restricoes.csv")
        return
    
    csv_file = sys.argv[1]
    
    # Se não for caminho absoluto, assumir que está na pasta atual
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(os.getcwd(), csv_file)
    
    plot_correlation(csv_file)

if __name__ == "__main__":
    main()