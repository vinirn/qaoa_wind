#!/bin/bash
#
# Script de Benchmarking QAOA - Turbinas Eólicas
# Executa múltiplas simulações variando um parâmetro específico
# 
# Uso:
#   ./benchmark_qaoa.sh <config_file> <parameter_path> <start_value> <end_value> <step_size> [num_runs_per_value]
#
# Exemplos:
#   ./benchmark_qaoa.sh config_3x3.json "qaoa.optimizer_options.rhobeg" 0.1 1.0 0.1
#   ./benchmark_qaoa.sh config_3x3.json "qaoa.layers" 1 3 1
#   ./benchmark_qaoa.sh config_3x3.json "wake_effects.base_penalty" 2.0 10.0 2.0 3
#

set -e

# Detectar flag --plot
PLOT_FLAG=""
ARGS=("$@")
NEW_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--plot" ]; then
        PLOT_FLAG="--plot"
    else
        NEW_ARGS+=("$arg")
    fi
done

# Recriar argumentos sem --plot
set -- "${NEW_ARGS[@]}"

# Verificar argumentos
if [ $# -lt 5 ] || [ $# -gt 6 ]; then
    echo "❌ Uso incorreto!"
    echo ""
    echo "📋 Uso:"
    echo "   $0 <config_file> <parameter_path> <start_value> <end_value> <step_size> [num_runs_per_value] [--plot]"
    echo ""
    echo "📝 Parâmetros:"
    echo "   config_file       - Arquivo de configuração (ex: config_3x3.json)"
    echo "   parameter_path    - Caminho do parâmetro (ex: qaoa.optimizer_options.rhobeg)"
    echo "   start_value       - Valor inicial"
    echo "   end_value         - Valor final"
    echo "   step_size         - Incremento entre valores"
    echo "   num_runs_per_value - Número de execuções por valor (padrão: 1)"
    echo "   --plot            - Gerar gráficos para cada execução (opcional)"
    echo ""
    echo "🎯 Exemplos:"
    echo "   $0 config_3x3.json 'qaoa.optimizer_options.rhobeg' 0.1 1.0 0.1"
    echo "   $0 config_3x3.json 'qaoa.layers' 1 3 1 --plot"
    echo "   $0 config_3x3.json 'wake_effects.base_penalty' 2.0 10.0 2.0 3 --plot"
    echo ""
    exit 1
fi

CONFIG_FILE="$1"
PARAMETER_PATH="$2"
START_VALUE="$3"
END_VALUE="$4"
STEP_SIZE="$5"
NUM_RUNS="${6:-1}"

# Verificar se o arquivo de configuração existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Erro: Arquivo de configuração '$CONFIG_FILE' não encontrado!"
    exit 1
fi

# Verificar se Python está disponível
if ! command -v python3 &> /dev/null; then
    echo "❌ Erro: 'python3' não está disponível!"
    exit 1
fi

# Criar backup do arquivo original
BACKUP_FILE="${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo "💾 Backup criado: $BACKUP_FILE"

# Função para restaurar configuração original
restore_config() {
    echo "🔄 Restaurando configuração original..."
    cp "$BACKUP_FILE" "$CONFIG_FILE"
}

# Função para atualizar parâmetro no JSON usando Python
update_parameter() {
    local param_path="$1"
    local new_value="$2"
    
    python3 << EOF
import json
import sys

# Carregar JSON
try:
    with open('$CONFIG_FILE', 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Erro ao ler arquivo JSON: {e}", file=sys.stderr)
    sys.exit(1)

# Navegar pelo caminho e atualizar valor
path_parts = '$param_path'.split('.')
obj = data

try:
    # Navegar até o objeto pai
    for part in path_parts[:-1]:
        obj = obj[part]
    
    # Detectar tipo do valor
    value = '$new_value'
    if value.replace('.', '', 1).replace('-', '', 1).isdigit():
        # Valor numérico
        if '.' in value:
            obj[path_parts[-1]] = float(value)
        else:
            obj[path_parts[-1]] = int(value)
    elif value.lower() in ['true', 'false']:
        # Valor booleano
        obj[path_parts[-1]] = value.lower() == 'true'
    else:
        # Valor string
        obj[path_parts[-1]] = value
    
    # Salvar arquivo atualizado
    with open('$CONFIG_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("OK")
    
except Exception as e:
    print(f"Erro ao atualizar parâmetro: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Função para gerar sequência de valores
generate_values() {
    python3 -c "
import sys
start, end, step = float('$START_VALUE'), float('$END_VALUE'), float('$STEP_SIZE')
current = start
while current <= end + 1e-10:  # Tolerância para ponto flutuante
    if current == int(current):
        print(int(current))
    else:
        print(f'{current:.10g}')
    current += step
"
}

# Configurar trap para restaurar configuração em caso de interrupção
trap restore_config EXIT

echo "=================================="
echo "🚀 BENCHMARKING QAOA - TURBINAS EÓLICAS"
echo "=================================="
echo ""
echo "📋 Configuração do Benchmark:"
echo "   • Arquivo de config: $CONFIG_FILE"
echo "   • Parâmetro: $PARAMETER_PATH"
if [ -n "$PLOT_FLAG" ]; then
    echo "   • Gráficos: ✅ ATIVADOS (--plot)"
else
    echo "   • Gráficos: ❌ DESATIVADOS"
fi
echo "   • Valores: $START_VALUE → $END_VALUE (passo: $STEP_SIZE)"
echo "   • Execuções por valor: $NUM_RUNS"
echo ""

# Obter valor atual do parâmetro
CURRENT_VALUE=$(python3 -c "
import json
try:
    with open('$CONFIG_FILE', 'r') as f:
        data = json.load(f)
    path_parts = '$PARAMETER_PATH'.split('.')
    obj = data
    for part in path_parts:
        obj = obj[part]
    print(obj)
except:
    print('null')
")
echo "📊 Valor atual de '$PARAMETER_PATH': $CURRENT_VALUE"
echo ""

# Gerar lista de valores para testar
VALUES=($(generate_values))
TOTAL_RUNS=$((${#VALUES[@]} * NUM_RUNS))

echo "🎯 Valores a serem testados: ${VALUES[*]}"
echo "📈 Total de execuções: $TOTAL_RUNS"
echo ""

# Contador de progresso
CURRENT_RUN=0

# Loop principal do benchmarking
for VALUE in "${VALUES[@]}"; do
    echo "========================================"
    echo "🔧 Testando $PARAMETER_PATH = $VALUE"
    echo "========================================"
    
    # Atualizar parâmetro no arquivo de configuração
    if ! update_parameter "$PARAMETER_PATH" "$VALUE"; then
        echo "❌ Erro ao atualizar parâmetro '$PARAMETER_PATH' para '$VALUE'"
        continue
    fi
    
    # Verificar se a atualização foi bem-sucedida
    UPDATED_VALUE=$(python3 -c "
import json
try:
    with open('$CONFIG_FILE', 'r') as f:
        data = json.load(f)
    path_parts = '$PARAMETER_PATH'.split('.')
    obj = data
    for part in path_parts:
        obj = obj[part]
    print(obj)
except:
    print('null')
")
    echo "✅ Parâmetro atualizado: $PARAMETER_PATH = $UPDATED_VALUE"
    echo ""
    
    # Executar múltiplas vezes se especificado
    for ((run=1; run<=NUM_RUNS; run++)); do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        if [ $NUM_RUNS -gt 1 ]; then
            echo "📊 Execução $run/$NUM_RUNS para $PARAMETER_PATH = $VALUE"
            echo "🏃‍♂️ Progresso geral: $CURRENT_RUN/$TOTAL_RUNS"
        else
            echo "🏃‍♂️ Progresso: $CURRENT_RUN/$TOTAL_RUNS"
        fi
        echo ""
        
        # Executar QAOA
        echo "⚡ Executando QAOA com $PARAMETER_PATH = $VALUE..."
        if [ -n "$PLOT_FLAG" ]; then
            echo "📊 Gerando gráficos (--plot ativado)"
            if ! ./run_qaoa.sh "$CONFIG_FILE" "$PLOT_FLAG"; then
                echo "❌ Erro na execução do QAOA para $PARAMETER_PATH = $VALUE (run $run)"
                # Continuar com próxima execução mesmo em caso de erro
            fi
        else
            if ! ./run_qaoa.sh "$CONFIG_FILE"; then
                echo "❌ Erro na execução do QAOA para $PARAMETER_PATH = $VALUE (run $run)"
                # Continuar com próxima execução mesmo em caso de erro
            fi
        fi
        echo ""
        
        # Pequena pausa entre execuções
        if [ $run -lt $NUM_RUNS ]; then
            sleep 1
        fi
    done
    
    echo ""
done

# Restaurar configuração original
restore_config

echo "========================================"
echo "✅ BENCHMARKING CONCLUÍDO!"
echo "========================================"
echo ""
echo "📊 Resumo:"
echo "   • Parâmetro testado: $PARAMETER_PATH"
echo "   • Valores testados: ${VALUES[*]}"
echo "   • Total de execuções: $TOTAL_RUNS"
echo "   • Configuração original restaurada"
echo ""
echo "📁 Resultados salvos automaticamente em:"
echo "   resultados_simulacoes/"
echo ""
echo "💡 Para analisar os resultados:"
echo "   • Abra o arquivo CSV correspondente"
echo "   • Compare os valores da coluna '$PARAMETER_PATH' vs métricas de performance"
echo "   • Analise execution_time_s, net_score, penalty_ratio, etc."
echo ""