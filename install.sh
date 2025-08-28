#!/bin/bash

echo "=== Instalação QAOA Turbinas Eólicas ==="
echo ""

# Detectar sistema operacional Linux e gerenciador de pacotes
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt &> /dev/null; then
        PKG_MANAGER="apt"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        echo "❌ Gerenciador de pacotes Linux não suportado. Este script suporta apt, yum ou dnf."
        exit 1
    fi
    echo "🖥️  Sistema Linux detectado com $PKG_MANAGER"
else
    echo "❌ Este script é destinado apenas para sistemas Linux."
    echo "Para outros sistemas, instale manualmente:"
    echo "  - Git: https://git-scm.com/downloads"
    echo "  - Python3: https://python.org/downloads"
    exit 1
fi

# Instalar Git se necessário
if ! command -v git &> /dev/null; then
    echo "🔧 Instalando Git..."
    case $PKG_MANAGER in
        "apt")
            sudo apt update && sudo apt install -y git
            ;;
        "yum")
            sudo yum install -y git
            ;;
        "dnf")
            sudo dnf install -y git
            ;;
    esac
    
    if command -v git &> /dev/null; then
        echo "✅ Git instalado com sucesso"
    else
        echo "❌ Erro ao instalar Git"
        exit 1
    fi
else
    echo "✅ Git já está instalado: $(git --version)"
fi

echo ""

# Instalar Python3 e venv se necessário
if ! command -v python3 &> /dev/null; then
    echo "🔧 Instalando Python3..."
    case $PKG_MANAGER in
        "apt")
            sudo apt update && sudo apt install -y python3 python3-pip python3-venv
            ;;
        "yum")
            sudo yum install -y python3 python3-pip python3-venv
            ;;
        "dnf")
            sudo dnf install -y python3 python3-pip python3-venv
            ;;
    esac
else
    # Verificar se venv está disponível
    if ! python3 -m venv --help &> /dev/null; then
        echo "🔧 Instalando python3-venv..."
        case $PKG_MANAGER in
            "apt")
                sudo apt update && sudo apt install -y python3-venv
                ;;
            "yum")
                sudo yum install -y python3-venv
                ;;
            "dnf")
                sudo dnf install -y python3-venv
                ;;
        esac
    fi
fi

echo "✅ Python3 encontrado: $(python3 --version)"
echo ""

# Criar ambiente virtual
echo "🔧 Criando ambiente virtual..."
if [ -d "qiskit_env" ]; then
    echo "⚠️  Ambiente virtual já existe. Removendo..."
    rm -rf qiskit_env
fi

python3 -m venv qiskit_env

if [ $? -eq 0 ]; then
    echo "✅ Ambiente virtual criado com sucesso"
else
    echo "❌ Erro ao criar ambiente virtual"
    exit 1
fi

echo ""

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source qiskit_env/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Ambiente virtual ativado"
else
    echo "❌ Erro ao ativar ambiente virtual"
    exit 1
fi

echo ""

# Verificar se requirements.txt existe
if [ ! -f "requirements.txt" ]; then
    echo "❌ Arquivo requirements.txt não encontrado"
    exit 1
fi

echo "🔧 Instalando dependências do requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependências instaladas com sucesso"
else
    echo "❌ Erro ao instalar dependências"
    exit 1
fi

echo ""
echo "🎉 INSTALAÇÃO CONCLUÍDA!"
echo ""
echo "Para usar o projeto:"
echo "  1. Ativar ambiente: source qiskit_env/bin/activate"
echo "  2. Executar QAOA: ./run_qaoa.sh"
echo "  3. Desativar ambiente: deactivate"
echo ""
echo "Exemplo de execução:"
echo "  ./run_qaoa.sh           # usa configuração embutida (hardcoded)"
echo ""
