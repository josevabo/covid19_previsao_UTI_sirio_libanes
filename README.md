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

O arquivo contendo os dados utilizados neste projeto se encontra no repositório deste projeto no Github (https://github.com/josevabo/covid19_previsao_UTI_sirio_libanes/blob/main/dados/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx?raw=true)

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
