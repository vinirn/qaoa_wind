# QAOA Wind — Turbinas Eólicas

O projeto implementa um QAOA (Qiskit 2.x) para otimizar a posição de turbinas eólicas em um grid simples, considerando penalidades de esteira (wake). A configuração agora é embutida (hardcoded) no código — arquivos JSON foram descontinuados.

## Requisitos
- Python 3.10+
- Ambiente virtual recomendado

## Instalação rápida
```
python3 -m venv qiskit_env && source qiskit_env/bin/activate
pip install -r requirements.txt
```

## Como executar
- Script (recomendado): `./run_qaoa.sh`
- Direto no Python: `source qiskit_env/bin/activate && python qaoa_turbinas.py`

Opções úteis:
- `--plot`: salva gráficos em `images/`
- `--plot-quantum`: plota o circuito QAOA
- `--ibm-quantum`: executa via IBM Quantum (requer credenciais locais)

Notas:
- A execução usa configuração embutida (2x3, vento oeste→leste). Parâmetros JSON e `-c/--config` são ignorados.

## Saídas
- CSV em `resultados_simulacoes/` com resumo da execução
- PNGs em `images/` quando `--plot` é usado

## IBM Quantum (opcional)
Para usar `--ibm-quantum`, crie dois arquivos locais (não versionados):

- `apikey.json`
```
{ "apikey": "SEU_TOKEN_IBM_CLOUD" }
```

- `ibm.json` (mínimo)
```
{
  "instance": "<CRN_DA_INSTANCIA>",
  "backends": { "primary": "ibm_torino", "fallback": "ibm_brisbane" },
  "transpilation": { "optimization_level": 3 },
  "plan": { "type": "Open", "monthly_limit": "10 minutos", "cost_per_shot": 0.0 }
}
```

Em seguida, execute com `./run_qaoa.sh --ibm-quantum`.
