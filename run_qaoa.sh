#!/bin/bash

# Script para executar QAOA em ambiente virtual
# Uso: ./run_qaoa.sh [arquivo_python]

echo "=== Script QAOA - Turbinas Eólicas ==="
echo ""

# Verificar se ambiente virtual existe
if [ ! -d "qiskit_env" ]; then
    echo "❌ Ambiente virtual 'qiskit_env' não encontrado!"
    echo "Execute primeiro: python3 -m venv qiskit_env"
    exit 1
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source qiskit_env/bin/activate

# Verificar se ativação foi bem-sucedida
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Falha ao ativar ambiente virtual!"
    exit 1
fi

echo "✅ Ambiente ativado: $VIRTUAL_ENV"
echo ""

# Determinar comando a executar
if [ "$1" = "" ]; then
    # Padrão: QAOA com config padrão
    COMMAND="python qaoa_turbinas.py"
    echo "🚀 Executando QAOA (configuração padrão)..."
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    python qaoa_turbinas.py --help
    exit 0
elif [ "$1" = "--list-configs" ]; then
    python qaoa_turbinas.py --list-configs
    exit 0
elif [[ "$1" == *.json ]]; then
    # Se primeiro argumento é um arquivo JSON, usar como configuração
    # Mas passar todos os demais argumentos também
    if [ $# -gt 1 ]; then
        COMMAND="python qaoa_turbinas.py -c $1 ${@:2}"
        echo "🚀 Executando QAOA com configuração: $1 e argumentos: ${@:2}"
    else
        COMMAND="python qaoa_turbinas.py -c $1"
        echo "🚀 Executando QAOA com configuração: $1"
    fi
else
    # Passar todos os argumentos para o script Python
    COMMAND="python qaoa_turbinas.py $*"
    echo "🚀 Executando QAOA com argumentos: $*"
fi

echo ""
echo "================================================"

# Executar o comando
$COMMAND

echo ""
echo "================================================"
echo "✅ Execução concluída!"
echo ""

# Desativar ambiente virtual
deactivate