# Execução no IBM Quantum

Este documento descreve como executar o QAOA de turbinas eólicas em computadores quânticos da IBM.

## Configuração de Custo Otimizada

Para o grid 3x3 (9 qubits), recomendamos:
- **Backend**: `ibm_peekskill` (27 qubits)
- **Shots**: 100 por iteração
- **Iterações**: 50 do otimizador
- **Custo estimado**: ~$4
- **Tempo**: 10min-1h (fila) + 3-8min (execução)

## Setup Necessário

### 1. Instalação
```bash
pip install qiskit-ibm-runtime
```

### 2. Configuração de Acesso
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Primeira vez - salvar credenciais
QiskitRuntimeService.save_account(token="seu_token_aqui")

# Uso posterior
service = QiskitRuntimeService()
backend = service.backend("ibm_peekskill")
```

### 3. Alterações no Código

Substituir no `qaoa_turbinas.py`:

```python
# Trocar:
from qiskit_aer.primitives import EstimatorV2

# Por:
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

# Na função run_qaoa():
service = QiskitRuntimeService()
backend = service.backend("ibm_peekskill")
estimator = EstimatorV2(backend=backend)
```

## Comparação de Backends

| Backend | Qubits | Custo/1000 shots | Total (~$) | Recomendação |
|---------|--------|------------------|------------|--------------|
| ibm_peekskill | 27 | $0.80 | ~$4 | ⭐ **Melhor custo-benefício** |
| ibm_torino | 133 | $0.80 | ~$4 | Alternativa |
| ibm_brisbane | 127 | $1.60 | ~$8 | Premium |
| ibm_kyoto | 127 | $1.60 | ~$8 | Premium |

## Otimizações para Hardware Real

1. **Error Mitigation**: Ativar se disponível
2. **Transpilação**: `optimization_level=3`
3. **Layout**: Deixar automático para melhor mapeamento
4. **Monitoramento**: Acompanhar fila em tempo real