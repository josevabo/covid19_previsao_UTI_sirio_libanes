# Previsão de internações por COVID19 - Hospital Sírio Libanês
![banner hospital](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/banner_hospital_2.jpg?raw=true)

# Objetivo

Este projeto se trata de um trabalho de conclusão do Bootcamp Data Science da Alura 2021. O tema proposto pelos instrutores do curso foi de atender a um desafio no site kaggle.com para uso se Machine Learning na predição de internação de pacientes com suspeita de Covid19.

Abaixo será apresentado o contexto e todo trabalho desenvolvido que está disponível no notebook presente neste repositório.

# Contexto: Pandemia e ações necessárias

Em 2020 o mundo foi surpreendido por uma nova pandemia. A COVID19 em pouco tempo mudou a vida de todos no mundo todo. Pessoas e empresas precisaram se mobilizar para encarar todas as mudanças e restrições que o combate à pandemia nos impôs.

Muitos mercados sofreram economicamente, vidas foram perdidas (e ainda são, no momento da escrita desse projeto), empresas faliram, massas perderam seus empregos ou sofreram financeiramente de alguma forma. Diante do caos causado pelo medo e insegurança desta nova doença, as ciências médicas foram postas à prova. Todo o mundo esperava ansiosamente por uma solução que viesse dos médicos, laboratórios farmacêuticos e demais cientistas.

O desenvolvimento de uma vacina, uma cura, tratamentos eficazes que levasse a zero a letalidade desta nova doença. Diante de tanta expectativa, hospitais como o Sírio Libanês investiram muito em pesquisas para encontrar formas mais eficazes de atender a população e oferecer o melhor tratamento possível.

Dado este contexto, neste projeto trataremos de uma ação do Hospital Sírio Libanês. O hospital propôs ao público o desafio de desenvolver um modelo de Machine Learning que pudesse prever possibilidade de internação a partir de dados de diagnósticos de pacientes. Esta ação foi proposta na plataforma voltada à comunidade de Data Science, Kaggle, no link abaixo:

https://www.kaggle.com/S%C3%ADrio-Libanes/covid19

# O problema: previsão de internações

Um fato percebido logo no início da pandemia e da explosão de casos mundialmente foi que a infecção pelo coronavírus levava muitas pessoas a necessitarem de atendimento em UTI, principalmente pela necessidade de respiradores mecânicos, já que a doença comumente levava a um comprometimento da capacidade pulmonar de seus portadores.

Com o compartilhamento de informações e conhecimento sobre a evolução da doença em outros países, um dos maiores perigos que assombrava qualquer governo e população era a possibilidade de sobrecarregar o sistema de saúde, com a alta demanda de leitos de UTI diante da perda de controle da curva de contágio pela doença. Este risco foi a principal justificativa por trás das medidas restritivas à circulação e aglomeração de pessoas adotadas no mundo inteiro.


|![Curva de casos COVID](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/curva_casos.jpg?raw=true)|
|:--:|
|<b>Simulações iniciais das curvas de casos de COVID comparando uma situação com contágio controlado e outra sem controle da curva de contágio</b>|

O risco de colapso nos sistemas de saúde exigiu mais do que nunca respostas rápidas dos profissionais de saúde. Como a demanda por UTIs corria o risco de chegar no limite, qualquer decisão envolvendo um leito de UTI é crítica. Neste cenário que a equipe do Sírio Libanês propõe o desafio citado anteriormente à comunidade de cientistas de dados no Kaggle: a partir de dados clínicos de pacientes que deram entrada no hospital, prever quem deve ser enviado à UTI ou não.

# Desafio Proposto

A partir de um conjunto de dados de pacientes que ingressaram no hospital com suspeita de COVID19, prever a necessidade de internação ou não de tal paciente. Os dados para cada paciente são dos tipos demográficos, histórico de doenças prévias, resultados de exame de sangue e sinais vitais.

Os dados oferecidos serão detalhados no próximo tópico.

**Diante do desafio, o objetivo deste projeto é, após a devida exploração dos dados oferecidos, treinar um modelo capaz de prever com a maior precisão possível a internação dos pacientes que ingressam no hospital com suspeita de COVID19.**

# Dados disponíveis

Os dados presentes no dataset contém registros anonimizados de pacientes que deram entrada nas unidades São Paulo e Brasília do Hospital Sírio-Libanês.

Os dados já passaram por uma limpeza inicial e normalização pela equipe do Hospital, apresentando valores de -1 a 1, nos campos que permitem normalização deste tipo.

## Estrutura do Dataset:

São 1925 registros, com 231 colunas.

 PACIENT_VISIT_IDENTIFIER | ... | WINDOW | ICU
---------------|----------|:-:|-:
 1 | ...| 0-2| 0
 1 | ...| 2-4| 1
 2 | ...| 0-2| 1
 3 | ...| 0-2| 0
...| ...|... |...
384|... | 4-6| 0

Cada registro representa uma janela de permanência de determinado paciente (sendo cada paciente único identificado pelo campo PATIENT_VISIT_IDENTIFIER), sendo 5 janelas possíveis (informadas no campo WINDOW):

| WINDOW | Período contido na janela |
|---------------|:----------:|
 | 0-2 | de 0 a 2 horas após a entrada|
 | 2-4 | de 2 a 4 horas após a entrada|
|  4-6 | de 4 a 6 horas após a entrada|
|  6-12 | de 6 a 12 horas após a entrada|
| Above_12 | acima de 12 horas após a entrada|

|![Janelas de permanência do paciente](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/windows.jpg?raw=true)|
|:--:|
|<b>Ilustração das janelas de permanência do paciente e o descarte dos dados de janelas pós internação. Fonte: Sírio-Líbanes - kaggle.com</b>|


As 231 colunas correspondem a 54 características diferentes. Porém, algumas características possuem outras colunas associadas contendo dados como variação deste indicador, máximo registrado no período da janela, mínimo registrado no perído, média, mediana e diferença relativa. 

Estas 54 características/features estão separadas em 4 subgrupos principais:
- Dados demográficos do paciente (3): ID, faixa etária e gênero;
- Doenças prévias detectadas (9): são 7 grupos de doenças anonimizados, pressão alta e imunocomprometimento;
- Resultados de exames de sangue (36): albuminas, por exemplo;
- Sinais vitais (6): frequência cardíaca, por exemplo;

# Limpeza dos dados

Os dados passaram por processos de limpeza conforme as etapas abaixo para garantir uma melhor normalização para posterior uso como dados de treino de um modelo de Machine Learning:

- Preenchimento de campos vazios com dados próximos existentes para as features contínuas**

- Remoção dos registros dos pacientes que possuem marcação de encaminhamento para UTI já na primeira janela (0-2)
  
- Eliminar linhas que ainda possuem dados vazios

- Manter apenas os dados da primeira janela (0 - 2 horas) e marcar os registros que resultaram em encaminhamento à UTI.

- Conversão da coluna AGE_PERCENTIL em dados categóricos

## Remoção de features/colunas altamente correlacionadas

Correlação indica a "semelhança" entre duas variáveis. Dados altamente correlacionados podem indicar uma redundância de features em nosso treino, por isso decidimos remover features que apresentem essa característica quando comparadas a outras.

Para definir uma linha de corte entre o que é alta correlação e não é desejada em nossos dados, e o que não é e pode ser mantida na base, **determinamos o valor limite para correlação como 0.95 **(de um limite teórico de 1).

A partir desse valor executamos um processo de limpeza que eliminou mais de 100 colunas que tinham alta correlação com outras que foram mantidas.

![Gráfico de correlação](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_corr.jpg?raw=true)

O gráfico acima mostra a matriz de correlação que utilizamos para visualizar de maneira mais intuitiva a correlação entre features em nossos dados. Quanto mais vermelho, maior a correlação. As setas mostram como um ponto no gráfico faz referência a duas features que estão sendo cruzadas.

# Exploração dos dados após limpeza

Após a limpeza inicial, analisamos algumas features mais livremente para vermos se conseguimos algum conhecimento maior na relação do problema com os nossos dados.

Para facilitar as análises montamos visualizações para cada contexto avaliado.

## Análise da de internação por faixas etárias

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_age_percentil.png?raw=true)

Observando os dados acima é perceptível que há um aumento na taxa de internação de pacientes conforme maior a faixa etária do mesmo. Na visualização as taxas são divididas por faixas de idade de 10 em 10 anos.

## Comparação de internação entre menores e maiores de 65 anos

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_ageabove.jpg?raw=true)

Como visto na comparação das taxas de internação por faixa etária, as internações são mais frequentes para pacientes maiores de 65 anos.

## Features com alta correlação com a ocorrência de internação

Na análise de correlação entre features do dataset pudemos listar e ordenar aquelas que possuem maior correlação com a variável objetivo, ICU.
Com esta lista podemos observar como a distribuição destas variáveis se comporta diante das ocorrências de internação.

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_features.png?raw=true)

Acima observamos as distribuições binárias para 6 features que figuraram no nosso top 10 daquelas com maior correlação com a variável ICU.

A maioria delas se trata de índices obtidos em exames de sangue que para leigos não traz informação muito clara. Mas existe a **RESPIRATORY_RATE_MEAN** entre elas, relacionada à taxa de respiração do paciente, considerando apenas o nome. **Observando sua distribuição, podemos inferir que em pacientes que foram futuramente internados, é mais frequente o comportamento de uma maior taxa média de respiração que nos demais**.

## Outras features de interesse

Após as análises anteriores, algumas colunas aparentemente de interesse do foram selecionadas arbitrariamente a fim de conhecer sua relação com o agravamento da doença ou não.

Para sua análise, vamos adotar um olhar por dois grupos: doenças pré-existentes e dados coletados em exames.

**Doenças pré-existentes:**

> IMMUNOCOMPROMISED, DISEASE GROUPING 1, DISEASE GROUPING 2, DISEASE GROUPING 3, DISEASE GROUPING 4, DISEASE GROUPING 5, DISEASE GROUPING 6

São 7 colunas que correspondem, cada uma, a um determinado grupo de doenças não informadas. Todas são variáveis categóricas, o que nos permite comparar o percentual de ocorrência de internações entre pacientes que portam as doenças ou não.

**Dados coletados em exames:**

> SAT02_ARTERIAL_MEDIAN, OXYGEN_SATURATION_MEAN

Duas colunas com dados contínuos, os quais analisamos suas distribuições entre pessoas internadas e não internadas.

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_outras_features.png?raw=true)

Acima vemos o percentual de internações nos pacientes com doenças pré-existentes, divididos para cada grupo de doenças no dataset.

Como esperado, para quase todas os grupos, é mais provável a internação de um paciente que porte doença pré-existente. A única exceção foi para o DISEASE GROUPING 6.bit_length

O grupo de doenças que apresentou maior relação com casos de internação foi o DISEASE GROUPING 4, no qual um paciente que se enquadre neste perfil de doenças tem quase o dobro de chances de ser internado.

# Modelagem de Machine Learning

Com os dados preparados para treino podemos escolher nossos modelos para iniciar treino e teste de Machine Learning.

Antes do treino dos modelos iniciais realizamos a separação dos dados de treino e teste para termos dados reais para testar o modelo treinado.

Como o problema aqui se trata de uma classificação binária, precisamos buscar os tipos de modelos que aderem a esse tipo de classificação.
Os modelos selecionados estão, em sua maioria, presentes na biblioteca `scikit-learn`. Inicialmente realizaremos o processo de `fitting` (treino) de cada modelo, 
após o treino realizamos teste do modelo e comparamos suas acurácias, que é a porcentagem de classificações acertadas no teste, sendo nossa primeira métrica para mensurar os modelos.

Os modelos testados inicialmente são:
> LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, LinearSVC, RandomForestClassifier e LogisticRegression


Os modelos de melhor performance nos testes iniciais foram:
 
- RandomForestClassifier
- LogisticRegression
- GradientBoostingClassifier
- XGBClassifier
 
Os modelos foram testados e observados sob a métrica AUC. Os resultados abaixo foram obtidos:
```
Modelo RandomForestClassifier
AUC médio 0.8006408925791384
--------------
Modelo LogisticRegression
AUC médio 0.7739854696419305
--------------
Modelo GradientBoostingClassifier
AUC médio 0.788365334717177
--------------
Modelo XGBClassifier
AUC médio 0.786533471717696
```

Os testes foram realizados utilizando processo iterativo de treinamento de vários modelos, cálculo da performance de cada um e ao final obtida a média de todos.

Após estes resultados, os modelos `RandomForestClassifier` e `XGBClassifier` foram selecionados para seguir adiante.

## Tuning dos modelos

Os modelos testados até agora foram na sua forma mais pura, sem ajustes nos seus parâmetros internos, os hiperparâmetros.
São diversos parâmetros que variam para cada tipo de modelo. E para cada parâmetro existe um intervalo de valores possíveis. Dito isto, o ajuste e teste iterativo de cada hiperparâmetro manualmente tomaria um tempo impraticável.

Para evitar o tuning manual, foram utilizados os processos de exploração de configurações de hiperparâmetros `RandomizedSearchCV` e `GridSearchCV`. Combinando ambos, podemos explorar diversas combinações de ajustes aleatoriamente dentro de intervalos definidos por nós. O número ďe tentativas é definido também para tomar o tempo que o usuário achar necessário. No nosso caso o processo foi realizado em 100 tentativas de ajustes diferentes.
Após a busca aleatória, temos uma dica dos melhores ajustes encontrados na busca, devido ao teste de performance que é realizado durante o processo automaticamente.

Essa busca do melhor ajuste foi realizado para os dois modelos selecionados anteriormente e os ajustes do melhor modelo foram os seguintes:

Para o `RandomForestClassifier`:

```
{'bootstrap': False,
 'max_depth': 50,
 'max_features': 'auto',
 'max_samples': 0.2,
 'min_samples_leaf': 4,
 'min_samples_split': 20,
 'n_estimators': 3000}
```

Para o `XGBClassifier`:

```
{'min_child_weight': 1, 
'n_estimators': 400, 
'subsample': 0.7}
```

Com estes ajustes deveríamos ter o modelo otimizado, segundo os processos de tuning que utilizamos.
Comparando os modelos ajustados com os melhores parâmetros buscados, temos a tabela abaixo:


| <h3>Modelo  | <h3>AUC médio |
|:--|:--|
|Random Forest ajustado|<b>0.7998|
|Random Forest Default|<b>0.7971|
|XGBoost ajustado| <b>0.7929|
|XGBoost default| <b>0.7881|

Os valores de AUC acima estão ordenados. E apesar de um pequeno ganho de performance sobre os modelos sem ajustes, o resultado ainda não é satisfatório diante do problema crítico que esse modelo deve atender.

# Conclusões

- Durante a análise exploratória dos dados foi percebido que alguns fatores considerados de risco amplamente divulgados são verdadeiros em nosso conjunto de dados.
  - Doenças pré-existentes identificadas nos pacientes apresentaram alta correlação com casos de internação. Em especial a hipertensão.
  - Um dos fatores mais relevantes para internação é a faixa etária do paciente. A taxa de internação entre pacientes já idosos foi aproximadamente o dobro da taxa para os mais jovens.
  - Taxas fisiológicas obtidas através de exames de sangue e alguns sinais vitais também se demonstraram importantes aliados na tomada da decisão para internação ou não dos pacientes. Isto nos lembra da importância de realizar um diagnóstico mais completo logo nas primeiras horas de atenção ao paciente.
 
- Após realização de testes com várias métricas, encontramos como melhores modelos os  RandomForestClassifier e XGBClassifier. 
 
- Ambos tiveram resultados próximos, mas se necessário escolher apenas um, nos testes finais o modelo RandomForest se manteve décimos de percento à frente.
 
- Realizado o processo de ajuste de hiperparâmetros, testamos modelos considerados ótimos em seus ajustes antes do treino. Os resultados obtidos foram pouco efetivos, em alguns momento perdendo performance quando comparados aos modelos sem ajustes.
 
- Abaixo segue resumo dos resultados finais:
 
| Modelo  | Score médio| AUC médio |
|:--|:--|:--|
|Random Forest ajustado|<b>0.7300</b>|<b>0.7998</b>|
|Random Forest Default|<b>0.7331</b>|<b>0.7971</b>|
|XGBoost ajustado| <b>0.7231</b>|<b>0.7929</b>|
|XGBoost default| <b>0.7220</b>|<b>0.7881</b>|
 
- Nestes resultados o modelo mais consistente foi o RandomForest ajustado. Porém, os resultados obtidos para a natureza crítica do problema ainda não são satisfatórios. 
 
- Apesar das performances abaixo dos 80%, os resultados obtidos mostram um bom caminho em direção ao objetivo, pois a quantidade de registros no dataset é fator fundamental para um modelo bem treinado e capaz de gneralizar seu aprendizado para lidar com os diversos cenários possíveis em produção. Portando, com um aumento na base de dados, seja pela disponibilização pela própria equipe do hospital ou pelo aumento do conjunto de dados de outra forma, o modelo tende a ficar cada vez mais robusto e generalista. Em um cenário como este é possível que o processo de tuning dos hiperparâmetros possa apresentar resultados mais efetivos.
 
- Uma sugestão para evolução deste projeto é o ajuste mais fino de hiperparâmetros em busca de melhor acurácia do modelo.
