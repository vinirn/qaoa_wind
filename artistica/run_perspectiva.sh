#!/bin/bash
# Script para renderizar configuração de turbinas lendo parâmetros de render_config.json

echo "=== Renderizando configuração de turbinas ==="

# Carregar configurações do JSON
CONFIG_FILE="render_config.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Erro: Arquivo de configuração não encontrado: $CONFIG_FILE"
    exit 1
fi

echo "📋 Carregando configurações de $CONFIG_FILE..."

# Verificar se jq está disponível
if ! command -v jq &> /dev/null; then
    echo "❌ Erro: jq não encontrado. Instale com: sudo apt-get install jq"
    exit 1
fi

# Extrair parâmetros do JSON usando jq
TERRENO=$(jq -r '.images.terreno' "$CONFIG_FILE")
TURBINA=$(jq -r '.images.turbina' "$CONFIG_FILE")
CSV=$(jq -r '.csv_config' "$CONFIG_FILE")

# Gera nome do arquivo de saída automaticamente a partir do CSV
# Lê número de linhas (ROWS) e colunas (COLS) do arquivo CSV
ROWS=$(awk 'NF{c++} END{print c+0}' "$CSV")
COLS=$(awk -F',' 'NR==1{print NF+0}' "$CSV")

# Diretório de saída
OUTPUT_DIR="grids_gerados"
mkdir -p "$OUTPUT_DIR"

if [[ "$ROWS" -gt 0 && "$COLS" -gt 0 ]]; then
    GRID_SIZE="${COLS}x${ROWS}"
    OUTPUT_BASENAME="grid_${GRID_SIZE}_$(date +%Y%m%d_%H%M%S).png"
    OUTPUT="$OUTPUT_DIR/$OUTPUT_BASENAME"
else
    # Fallback seguro caso CSV esteja vazio ou inválido
    OUTPUT_BASENAME="grid_$(date +%Y%m%d_%H%M%S).png"
    OUTPUT="$OUTPUT_DIR/$OUTPUT_BASENAME"
fi

# Tamanho será calculado automaticamente pelo render_terreno.py
SIZE="512,512"  # Valor mínimo, será expandido automaticamente

TERRENO_CENTER_X=$(jq -r '.terreno_center[0]' "$CONFIG_FILE")
TERRENO_CENTER_Y=$(jq -r '.terreno_center[1]' "$CONFIG_FILE")
TERRENO_CENTER="${TERRENO_CENTER_X},${TERRENO_CENTER_Y}"

TURBINA_BASE_X=$(jq -r '.turbina_base[0]' "$CONFIG_FILE")
TURBINA_BASE_Y=$(jq -r '.turbina_base[1]' "$CONFIG_FILE")
TURBINA_BASE="${TURBINA_BASE_X},${TURBINA_BASE_Y}"

# Deslocamentos
SHIFT_X=$(jq -r '.positioning.shift_x' "$CONFIG_FILE")
SHIFT_Y=$(jq -r '.positioning.shift_y' "$CONFIG_FILE")
SHIFT_ROW_X=$(jq -r '.positioning.shift_row_x' "$CONFIG_FILE")
SHIFT_ROW_Y=$(jq -r '.positioning.shift_row_y' "$CONFIG_FILE")

# Verificar se arquivos existem
if [[ ! -f "$TERRENO" ]]; then
    echo "❌ Erro: Arquivo de terreno não encontrado: $TERRENO"
    exit 1
fi

if [[ ! -f "$TURBINA" ]]; then
    echo "❌ Erro: Arquivo de turbina não encontrado: $TURBINA"
    exit 1
fi

if [[ ! -f "$CSV" ]]; then
    echo "❌ Erro: Arquivo de configuração não encontrado: $CSV"
    echo "Criando arquivo exemplo..."
    cat > "$CSV" << EOF
1,0,1
0,1,0
1,0,1
EOF
    echo "✅ Arquivo $CSV criado com configuração de exemplo"
fi

echo "🔧 Configurações carregadas de $CONFIG_FILE:"
echo "   • Terreno: $TERRENO"
echo "   • Turbina: $TURBINA"  
echo "   • Config: $CSV"
echo "   • Output: $OUTPUT (gerado automaticamente)"
echo "   • Centro terreno: $TERRENO_CENTER"
echo "   • Base turbina: $TURBINA_BASE"
echo "   • Shifts: x=$SHIFT_X, y=$SHIFT_Y, row_x=$SHIFT_ROW_X, row_y=$SHIFT_ROW_Y"
echo "   • Tamanho: Calculado automaticamente"

echo
echo "🚀 Executando renderização..."

python render_terreno.py \
    --csv "$CSV" \
    --terreno "$TERRENO" \
    --turbina "$TURBINA" \
    --output "$OUTPUT" \
    --size "$SIZE" \
    --terreno-center "$TERRENO_CENTER" \
    --turbina-base "$TURBINA_BASE" \
    --shift-x "$SHIFT_X" \
    --shift-y "$SHIFT_Y" \
    --shift-row-x "$SHIFT_ROW_X" \
    --shift-row-y "$SHIFT_ROW_Y"

if [[ $? -eq 0 ]]; then
    echo "✅ Renderização concluída!"
    echo "📁 Arquivo gerado: $OUTPUT"
    
    if command -v xdg-open &> /dev/null; then
        echo "🖼️  Abrindo imagem..."
        xdg-open "$OUTPUT"
    fi
else
    echo "❌ Erro na renderização"
    exit 1
fi
