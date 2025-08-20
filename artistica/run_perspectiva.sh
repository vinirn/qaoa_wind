#!/bin/bash
# Script para renderizar configuração de turbinas com parâmetros padrão

echo "=== Renderizando configuração de turbinas ==="

# Parâmetros padrão otimizados
TERRENO="terreno1_512x512.png"
TURBINA="turbina1_512x512.png"
CSV="config3x3.csv"
OUTPUT="saida_config3x3.png"

# Coordenadas e deslocamentos calibrados
TERRENO_CENTER="400,400"
TURBINA_BASE="250,200"
SHIFT_X=200.0
SHIFT_Y=150.0
SHIFT_ROW_X=40.0
SHIFT_ROW_Y=0.0
SIZE="1024,1024"

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

echo "🔧 Configurações:"
echo "   • Terreno: $TERRENO"
echo "   • Turbina: $TURBINA"  
echo "   • Config: $CSV"
echo "   • Output: $OUTPUT"
echo "   • Centro terreno: $TERRENO_CENTER"
echo "   • Base turbina: $TURBINA_BASE"
echo "   • Shifts: x=$SHIFT_X, y=$SHIFT_Y, row_x=$SHIFT_ROW_X"
echo "   • Tamanho: $SIZE"

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