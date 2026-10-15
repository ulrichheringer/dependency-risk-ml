# dependency-risk-ml

Estima risco de regressão em atualizações de dependências com sinais de versão, impacto e cobertura de testes.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout estratificado de 400 atualizações sintéticas — acurácia **0,8975**, precisão **0,8844**, recall **0,8793**, F1 **0,8818**, ROC AUC **0,9706**. O rótulo deriva de uma regra probabilística sobre os próprios sinais gerados; estes números não medem risco em pacotes reais.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
