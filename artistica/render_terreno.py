#!/usr/bin/env python3
"""
Script para renderizar terreno com turbinas posicionadas em grid
"""
import json
import csv
from pathlib import Path
from PIL import Image
import argparse

def load_positions_from_csv(csv_path):
    """Carrega posições das turbinas de um arquivo CSV (matriz de 0s e 1s) com transposição e flip vertical"""
    positions = []
    rows = 0
    cols = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            rows = row_idx + 1
            if row_idx == 0:
                cols = len(row)  # Define cols based on first row
            
            for col_idx, cell in enumerate(row):
                try:
                    if int(cell) == 1:
                        # Transposta: troca row e col -> (col_idx, row_idx)
                        # Depois flip vertical: nova_linha = cols - 1 - col_idx
                        flipped_row = cols - 1 - col_idx
                        positions.append((flipped_row, row_idx))
                except ValueError:
                    continue  # Skip invalid cells
    
    print(f"CSV carregado com transposição e flip vertical: {rows}x{cols} -> {cols}x{rows}")
    return positions, cols, rows  # Retorna cols, rows trocados também

def load_config(config_path):
    """Carrega configuração do arquivo JSON"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_tiling(terreno_path, turbina_path, positions, output_path, 
                 grid_size=(512, 512), terreno_center=(240, 240), turbina_base=(250, 380),
                 shift_x=0, shift_y=0, shift_row_x=0, shift_row_y=0):
    """
    Cria composição com turbinas posicionadas no terreno
    
    Args:
        terreno_path: Caminho para imagem do terreno
        turbina_path: Caminho para imagem da turbina
        positions: Lista de posições (row, col) onde colocar turbinas
        output_path: Caminho para salvar a imagem final
        grid_size: Tamanho do grid em pixels
        terreno_center: Centro do terreno em pixels (x, y)
        turbina_base: Centro da base da turbina em pixels (x, y)
        shift_x, shift_y: Deslocamento geral
        shift_row_x, shift_row_y: Deslocamento adicional por linha
    """
    # Carrega imagens
    terreno_original = Image.open(terreno_path).convert("RGBA")
    turbina = Image.open(turbina_path).convert("RGBA")
    
    # Redimensiona terreno para ser menor que as turbinas (para permitir grid)
    terrain_scale = 1.0  # Mantém tamanho original do terreno
    new_terrain_size = (int(terreno_original.size[0] * terrain_scale), int(terreno_original.size[1] * terrain_scale))
    terreno = terreno_original.resize(new_terrain_size, Image.Resampling.LANCZOS)
    
    # Calcula dimensões do grid baseado nas posições
    if not positions:
        print("Nenhuma posição especificada")
        return
        
    max_row = max(pos[0] for pos in positions)
    max_col = max(pos[1] for pos in positions)
    grid_rows = max_row + 1
    grid_cols = max_col + 1
    
    # Calcula tamanho automático baseado no grid e shifts
    # Espaço necessário = posição do último elemento + tamanho da imagem
    max_terrain_x = terreno_center[0] + (grid_cols - 1) * shift_x + (grid_rows - 1) * shift_row_x + terreno.size[0]
    max_terrain_y = terreno_center[1] + (grid_cols - 1) * shift_y + (grid_rows - 1) * shift_row_y + terreno.size[1]
    
    # Adiciona margem de segurança para as turbinas
    margin_x = turbina.size[0] // 2
    margin_y = turbina.size[1] // 8  # Reduzido ainda mais
    
    auto_width = int(max_terrain_x + margin_x)
    auto_height = int(max_terrain_y + margin_y)
    
    # Usa tamanho automático se for maior que o especificado
    final_width = max(grid_size[0], auto_width)
    final_height = max(grid_size[1], auto_height)
    
    print(f"Grid: {grid_rows}x{grid_cols}, Tamanho original: {grid_size[0]}x{grid_size[1]}")
    print(f"Tamanho calculado: {auto_width}x{auto_height}, Final: {final_width}x{final_height}")
    
    # Cria imagem final vazia
    result = Image.new("RGBA", (final_width, final_height), (0, 0, 0, 0))
    
    
    print(terreno_center)
    
    # Primeiro, posiciona blocos de terreno em todas as posições do grid
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calcula posição do centro do terreno para esta célula  
            center_x = terreno_center[0] + col * shift_x + row * shift_row_x
            center_y = terreno_center[1] + col * shift_y + row * shift_row_y
            
            # Posiciona terreno centralizado
            terreno_x = center_x 
            terreno_y = center_y 
            
            # Cola o bloco de terreno
            result.paste(terreno, (int(terreno_x), int(terreno_y)), terreno)
            print(f"Terreno ({row},{col}) -> pixel({terreno_x},{terreno_y})")
    
    # Depois, posiciona turbinas centralizadas nos blocos de terreno correspondentes
    for row, col in positions:
        # Usa EXATAMENTE a mesma lógica dos terrenos para calcular o centro
        center_x = terreno_center[0] + col * shift_x + row * shift_row_x
        center_y = terreno_center[1] + col * shift_y + row * shift_row_y
        
        # Calcula a posição do terreno correspondente (igual ao loop anterior)
        terreno_x_calc = center_x
        terreno_y_calc = center_y
                      
        # Posiciona a base da turbina no centro REAL do terreno
        turbina_x = terreno_x_calc + turbina_base[0]
        turbina_y = terreno_y_calc + turbina_base[1]
        
        # Cola a turbina
        result.paste(turbina, (int(turbina_x), int(turbina_y)), turbina)
        print(f"Turbina ({row},{col}) -> pixel({turbina_x},{turbina_y}) [centralizada no terreno]")
    
    # Gera nome do arquivo baseado na configuração do grid
    if output_path == "saida_config.png":  # Nome padrão
        grid_config = f"{cols}x{rows}"  # Após transposição
        num_turbines = len(positions)
        output_path = f"grid_{grid_config}_{num_turbines}turbinas.png"
    
    # Salva resultado
    result.save(output_path)
    print(f"Imagem salva em: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Renderiza terreno com turbinas em grid")
    parser.add_argument("--terreno", default="terreno1_512x512.png", 
                       help="Imagem do terreno (padrão: terreno1_512x512.png)")
    parser.add_argument("--turbina", default="turbina1_512x512.png",
                       help="Imagem da turbina (padrão: turbina1_512x512.png)")
    parser.add_argument("--csv", help="Arquivo CSV com posições das turbinas (formato: row,col)")
    parser.add_argument("--config", help="Arquivo JSON com configuração do grid")
    parser.add_argument("--positions", help="Posições das turbinas como 'row,col row,col ...'")
    parser.add_argument("--output", default="saida_config.png",
                       help="Arquivo de saída (padrão: saida_config.png)")
    parser.add_argument("--size", default="512,512", 
                       help="Tamanho da imagem final (padrão: 512,512)")
    parser.add_argument("--terreno-center", default="240,240",
                       help="Centro do terreno em pixels (padrão: 240,240)")
    parser.add_argument("--turbina-base", default="250,380", 
                       help="Centro da base da turbina (padrão: 250,380)")
    parser.add_argument("--shift-x", type=float, default=80.0,
                       help="Espaçamento horizontal do grid (padrão: 80)")
    parser.add_argument("--shift-y", type=float, default=60.0,
                       help="Espaçamento vertical do grid (padrão: 60)")
    parser.add_argument("--shift-row-x", type=float, default=40.0,
                       help="Deslocamento horizontal por linha (padrão: 40)")
    parser.add_argument("--shift-row-y", type=float, default=0.0,
                       help="Deslocamento vertical por linha (padrão: 0)")
    
    args = parser.parse_args()
    
    # Parse dos parâmetros
    width, height = map(int, args.size.split(','))
    grid_size = (width, height)
    
    terreno_center = tuple(map(int, args.terreno_center.split(',')))
    turbina_base = tuple(map(int, args.turbina_base.split(',')))
    
    # Determina posições das turbinas
    positions = []
    
    if args.csv:
        # Carrega de arquivo CSV
        try:
            positions, csv_rows, csv_cols = load_positions_from_csv(args.csv)
            print(f"Grid detectado: {csv_rows}x{csv_cols}")
            print(f"Carregadas {len(positions)} posições do arquivo CSV {args.csv}")
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return
    elif args.config:
        # Carrega de arquivo JSON
        try:
            config = load_config(args.config)
            positions = config.get("positions", [])
            print(f"Carregadas {len(positions)} posições do arquivo {args.config}")
            
            # Usar parâmetros do JSON se disponíveis
            if "terreno_center" in config:
                terreno_center = tuple(config["terreno_center"])
            if "turbina_base" in config:
                turbina_base = tuple(config["turbina_base"])
            if "positioning" in config:
                pos_config = config["positioning"]
                args.shift_x = pos_config.get("shift_x", args.shift_x)
                args.shift_y = pos_config.get("shift_y", args.shift_y)
                args.shift_row_x = pos_config.get("shift_row_x", args.shift_row_x)
                args.shift_row_y = pos_config.get("shift_row_y", args.shift_row_y)
                
        except Exception as e:
            print(f"Erro ao carregar config: {e}")
            return
    elif args.positions:
        # Parse manual das posições
        pos_strings = args.positions.split()
        for pos_str in pos_strings:
            row, col = map(int, pos_str.split(','))
            positions.append((row, col))
        print(f"Usando {len(positions)} posições: {positions}")
    else:
        # Exemplo padrão - grid 2x3
        positions = [(0,0), (0,2), (1,1)]
        print("Usando posições padrão:", positions)
    
    # Cria a composição
    create_tiling(args.terreno, args.turbina, positions, args.output, grid_size,
                 terreno_center, turbina_base, 
                 args.shift_x, args.shift_y, args.shift_row_x, args.shift_row_y)

if __name__ == "__main__":
    main()
