# Previsão de internações por COVID19 - Hospital Sírio Libanês
![banner hospital](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/banner_hospital_2.jpg?raw=true)

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

O arquivo contendo os dados utilizados neste projeto se encontra no repositório deste projeto no Github (https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/dados/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx?raw=true)

# Limpeza dos dados

Abaixo, com o auxílio das funções definidas nas células acima, são realizadas algumas etapas para limpeza dos dados iniciais que garantirão uma melhor normalização dos dados para posterior uso como dados de treino de um modelo de Machine Learning:

- **Preenchimento de campos vazios com dados próximos existentes para as features contínuas**
  - Para um treino adequado de um modelo de ML, é importante que não haja dados distorcidos no meio do dataset. Uma distorção que pode ocorrer é a existência de campos vazios. Em alguns casos esses dados nulos podem não ser bem interpretados no treino do modelo. Preencher campos numéricos com 0 nem sempre é a melhor opção, pois dependendo da natureza da feature, o valor zero ocupando um registro antes vazio entre valores positivos ou negativos já preenchidos anteriormente pode levar a um entendimento incorreto do cenário que esses dados em conjunto venham a mostrar.
  - Diante disso, a estratégia aqui adotada é a de **replicar nos campos vazios das features contínuas os dados já presentes nos registros próximos da mesma coluna**. A atenção especial com as features contínuas nesse caso se deve ao fato delas representarem no nosso dataset o intervalo de dados que representam resultados de exames e sinais vitais, podendo ser substituidas pelo valor coletado na janela seguinte ou anterior. 
  - Um cuidado importante nesta estratégia é evitar replicar dados de contextos completamente diferentes para preencher um dado vazio. Para evitar isso, primeiro agrupamos os registros pelo identificador do paciente e depois aplicar a replicação de dados próximos para os campos vazios. Assim evitamos que a o dado de frequência cardíaca vazio de um paciente "A" seja preenchido com o dado coletado para o paciente "B", por exemplo.
  - Isto é possível usando o método `fillna` da biblioteca pandas após o agrupamento. Este método nos permite preencher campos vazios, entre outras formas, replicando o mesmo valor presente no registro seguinte na mesma coluna, ou replicar o valor do registro anterior.

- **Remoção dos registros dos pacientes que possuem marcação de encaminhamento para UTI já na primeira janela (0-2)**
  - Seguindo recomendação da própria equipe que preparou os dados iniciais e o desafio no Kaggle, dados que possam ter sido coletados após o encaminhamento do paciente para UTI não devem ser utilizados para treino. Pois o objetivo é prever pacientes que possivelmente irão para UTI considerando os dados coletados antes que o evento ingresso em UTI ocorra.
  
- **Eliminar linhas que ainda possuem dados vazios**
  - Se após as etapas anteriores algum registro de paciente ainda possuir campos vazios, esta linha será eliminada para evitar uma distorção na hora do treino.

- **Manter apenas os dados da primeira janela (0 - 2 horas) e marcar os registros que resultaram em encaminhamento à UTI.**
  - Como o objetivo é um modelo que seja capaz de prever o quanto antes o risco de internação de um paciente, o ideal é termos um modelo capaz de já enxergar esse risco nas primeiras horas.
  - Para o correto treino do modelo, o paciente que acabou sendo encaminhado à UTI será marcado como ICU=1 no dataset. E todo dataset se tratará apenas de registros da primeira janela dos pacientes.

- **Conversão da coluna AGE_PERCENTIL em dados categóricos**
  - A coluna AGE_PERCENTIL apresenta uma informação que pode ser relevante e interessante para ser mantida nos dados de treino: a faixa etária do paciente. O problema é que os dados desta coluna são apresentados como texto ("60th","80th", etc), e dados textuais não conseguem ser usados como features em modelos de regressão, por exemplo. Para isso transformamos as categorias de faixas etárias apresentadas em texto para categorias numéricas usando o método `astype('category')` do pandas.

## Remoção de features/colunas altamente correlacionadas

Para definir uma linha de corte entre o que é alta correlação e não é desejada em nossos dados, e o que não é e pode ser mantida na base, **determinamos o valor limite para correlação como 0.95 **(de um limite teórico de 1).

Abaixo transformaremos os dados de forma que quaisquer colunas que tenham um coeficiente de correlação maior que 0.95 sejam eliminados do dataset.

Porém, se adotarmos o processo descrito acima, não eliminaremos somente as possíveis "duplicatas", mas também os dados de referência aos quais comparávamos a segunda coluna para identificar a alta correlação. Ou seja, se uma coluna que apresenta dados de frequência cardíaca demonstra alta correlação com a coluna de pressão sanguínea, as duas tem alta correlação entre si, portanto sendo excluídas se apenas considerarmos as colunas que apresentarem correlação maior que 0.95. Mas essa abordagem elimina dados que podem ser importantes. O procedimento adequado é eliminar apenas uma das colunas, assim reduzindo a possível ambiguidade.


![Gráfico de correlação](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_corr.jpg?raw=true)

Na imagem acima fica mais claro o exemplo anterior. Quando verificamos um ponto na matriz que indique correlação maior que 0.95 (seta amarela), este ponto faz referência a duas features (setas vermelhas). Ao eliminarmos apenas uma das duas features, já estamos removendo a alta correlação do dataset.

Para realizar esta remoção, utilizaremos de uma função que percorre a matriz de correlação analisando seus valores, onde houver valor maior que 0.95 a feature referente à coluna da matriz será eliminada, dessa forma a feature representada pela linha da matriz onde o ponto se encontra é mantida. Dessa forma, quando duas features forem identificadas como altamente correlacionadas, apenas uma delas será eliminada do conjunto de dados.

# Exploração dos dados após limpeza

Após a limpeza inicial e eliminação de features altamente correlacionadas, vamos observar algumas features mais livremente para vermos se conseguimos algum conhecimento maior na relação do problema com os nossos dados.

### Taxa de internação por faixa etária (faixas de 10 em 10 anos).

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_age_percentil.png?raw=true)
Observando os dados acima fica claro que há uma relação direta entre a ocorrência de internações e o aumento das faixas etárias. 

### Taxa de internação entre menores e maiores de 65 anos
![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_ageabove.jpg?raw=true)

Como visto na comparação das taxas de internação por faixa etária, **as internações são mais frequentes para pacientes maiores de 65 anos**.

### Features com alta correlação com a ocorrência de internação

Além das features de interesse vistas acima, observamos também a ocorrência de internações para outras features que apresentaram maior correlação com marcação de internação.

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_features.png?raw=true)

Acima observamos as distribuições binárias para as demais 6 features que figuraram no nosso top 10 daquelas com maior correlação com a variável ICU.

A maioria delas se trata de índices obtidos em exames de sangue que para leigos não traz informação muito clara. Mas existe a **RESPIRATORY_RATE_MEAN** entre elas, relacionada à taxa de respiração do paciente, considerando apenas o nome. **Observando sua distribuição, podemos inferir que em pacientes que foram futuramente internados, é mais frequente o comportamento de uma maior taxa média de respiração que nos demais**.

### Outras features de interesse

Algumas colunas identificadas visualmente parecem interessantes de serem analisadas a fim de conhecer sua relação com o agravamento da doença ou não.

As escolhi diretamente do dataset resultante da limpeza. Para sua análise, vamos adotar um olhar por dois grupos: doenças pré-existentes e dados coletados em exames.

**Doenças pré-existentes:**

> IMMUNOCOMPROMISED, DISEASE GROUPING 1, DISEASE GROUPING 2, DISEASE GROUPING 3, DISEASE GROUPING 4, DISEASE GROUPING 5, DISEASE GROUPING 6

São 7 colunas que correspondem, cada uma, a um determinado grupo de doenças não informadas. Todas são variáveis categóricas, o que nos permite comparar o percentual de ocorrência de internações entre pacientes que portam as doenças ou não.

**Dados coletados em exames:**

> SAT02_ARTERIAL_MEDIAN, OXYGEN_SATURATION_MEAN

Duas colunas com dados contínuos, os quais analisaremos suas distribuições entre pessoas internadas e não internadas.

![](https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/images/graf_outras_features.png?raw=true)

Acima vemos o percentual de internações nos pacientes com doenças pré-existentes, divididos para cada grupo de doenças no dataset.

Como esperado, para quase todas os grupos, é mais provável a internação de um paciente que porte doença pré-existente. A única exceção foi para o DISEASE GROUPING 6.bit_length

**O grupo de doenças que apresentou maior relação com casos de internação foi o DISEASE GROUPING 4, no qual um paciente que se enquadre neste perfil de doenças tem quase o dobro de chances de ser internado**.

# Modelagem de Machine Learning

Com os dados preparados e explorados
