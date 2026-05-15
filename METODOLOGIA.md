# Metodologia — modelo_bc_copulas_signals

## Objetivo

O projeto tem como objetivo estudar a geração de cenários sintéticos de retornos para ativos de renda variável via Cópula Gaussiana com marginal empírica, e utilizá-los para testar uma estratégia simples de mean-reversion em dados simulados. Os ativos analisados são quatro ações do mercado brasileiro: BBDC4, ITUB4, PETR4 e VALE3, com série histórica de preços ajustados de janeiro de 2008 a maio de 2026.

---

## 1. Dados

Utiliza-se a série de preços de fechamento ajustados, convertida em retornos simples diários. A partir desses retornos, todo o processo de estimação e simulação é conduzido.

---

## 2. Estimação de Volatilidade — EWMA

A volatilidade de cada ativo é estimada pelo modelo EWMA (Exponentially Weighted Moving Average), com parâmetro de decaimento `λ = 0.94`, padrão RiskMetrics. O comprimento efetivo da janela é determinado pelo parâmetro `ewma_conf = 0.9999`, que define até que ponto a massa acumulada de pesos é considerada representativa.

A volatilidade no instante `t` é dada por:

```
σ²_t = (1-λ) Σ λⁱ r²_{t-i}
```

A janela resultante tem aproximadamente 148 observações. Os pesos decaem exponencialmente, atribuindo maior relevância aos retornos mais recentes.

Com as volatilidades estimadas, os retornos históricos são padronizados (retornos ajustados por vol), removendo a heteroscedasticidade da série antes de construir a distribuição marginal.

---

## 3. Geração de Cenários — Cópula Gaussiana com Margem Empírica

O núcleo do modelo é a geração de caminhos futuros de retornos via Cópula Gaussiana. O processo a cada passo de tempo `t` da simulação é:

**a) Estado de volatilidade**  
A matriz de covariância dos retornos é mantida como estado evolutivo via EWMA multivariado:
```
Cov_t = λ · Cov_{t-1} + (1-λ) · r_{t-1} · r_{t-1}ᵀ
```
A volatilidade marginal de cada ativo é `σ_t = sqrt(diag(Cov_t))`.

**b) Estado de correlação**  
De forma análoga, uma pseudo-matriz de correlação `Q` é mantida com base nos resíduos padronizados `ε = r / σ`:
```
Q_t = λ · Q_{t-1} + (1-λ) · ε_{t-1} · ε_{t-1}ᵀ
```
A correlação corrente é extraída por `Corr_t = Q_t / outer(diag(Q_t)^{1/2})`, e a decomposição de Cholesky é aplicada sobre ela.

**c) Amostragem pela cópula**  
1. Gera-se um vetor de normais padrão correlacionadas via Cholesky: `z = L · N(0,I)`
2. Transforma-se em uniformes pela CDF normal: `u = Φ(z)`
3. Aplica-se a função quantílica empírica: cada `u_j` é mapeado para o quantil correspondente da distribuição histórica de retornos ajustados por vol do ativo `j`, reescalada pela volatilidade corrente

O resultado é um vetor de retornos simulados que:
- Respeita a estrutura de dependência cross-asset capturada pela correlação EWMA
- Reproduz as caudas e assimetrias históricas de cada ativo via margem empírica
- É condicionado ao nível de volatilidade corrente do mercado

Esse processo é repetido para cada um dos `T = 20` passos de tempo, atualizando `Cov` e `Q` a cada nova observação gerada. O modelo produz 10.000 caminhos independentes.

---

## 4. Geração do Sinal

O sinal de entrada na posição é baseado em um indicador de momentum negativo extremo sobre os retornos ajustados por volatilidade. Define-se uma janela de `REVERSION_WINDOW = 5` dias, e o sinal dispara quando a soma cumulativa dos retornos padronizados nessa janela está abaixo do percentil 5% de uma distribuição `N(0, √5)`:

```
threshold = Φ⁻¹(0.05; σ = √5) ≈ -3.68
```

A lógica intuitiva é: quando a performance ajustada a risco dos últimos 5 dias é extremamente negativa, há expectativa de recuperação. Para cada caminho simulado, registra-se apenas a primeira ocorrência do sinal por ativo, e o retorno cumulativo dos 5 dias subsequentes é capturado como resultado da estratégia.

A estacionariedade da estatística do sinal (soma rolling de retornos padronizados) é verificada sobre os dados históricos via testes ADF e KPSS.

---

## 5. Análise dos Resultados

Os resultados são avaliados via:
- Retorno médio e desvio-padrão das ocorrências do sinal, por ativo e agregado
- t-test unilateral (`H₀: retorno médio = 0`) e teste binomial sobre win rate (`H₀: p = 50%`)

Os resultados observados na simulação mostram retorno médio agregado de ~32 bps em 5 dias, com t-stat de 4.96.

---

## Observações

- O resultado positivo observado é resultado de um ambiente de estudo em dados sintéticos. Não há backtest em dados reais nem validação out-of-sample.
- O threshold do sinal assume normalidade das caudas do indicador, o que é uma aproximação.
- O t-test agrega observações de ativos correlacionados, o que infla nominalmente o tamanho da amostra.
- Custos de transação não estão incorporados.