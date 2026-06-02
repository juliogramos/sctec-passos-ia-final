# Desafio Extra - Hotel Booking Demand

Desafio extra do curso "Introdução à Inteligência Artificial" do programa [SCTEC](https://sctec.scti.sc.gov.br/).

Elaborado por [Julio Gonçalves Ramos](https://www.linkedin.com/in/julio-ramos-1684a5390/).

Link para o repositório: [https://github.com/juliogramos/sctec-passos-ia-final](https://github.com/juliogramos/sctec-passos-ia-final)

Link para o Notebook: [https://colab.research.google.com/drive/1Ybww0HESAxJa87vBnbPjoS0WE38JY0u7?usp=sharing](https://colab.research.google.com/drive/1Ybww0HESAxJa87vBnbPjoS0WE38JY0u7?usp=sharing)

## Tecnologias

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Como visualizar localmente

### Executar Notebook

1. Acessar o notebook: [link](https://colab.research.google.com/drive/1Ybww0HESAxJa87vBnbPjoS0WE38JY0u7?usp=sharing)
2. Executar células em ordem OU usar o botão Run All
   Não é preciso baixar o dataset manualmente, ele é baixado através da biblioteca do Kagglehub e salvo em cache.

### Executar arquivo Python

1. Clonar o repositório ou baixar os arquivos e extrair em uma pasta
2. Baixar o dataset: [link](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
3. Colocar o arquivo CSV na mesma pasta que o arquivo main.py e utilitarios.py
4. Instalar o Python (versão utilizada: 3.12.3) e o Pip
5. Instalar a biblioteca virtualenv com o comando: pip3 install virtualenv
6. Criar um novo ambiente virtual com o comando: virtualenv venv
7. Ativar o ambiente virtual com o comando (Mac e Linux): source venv/bin/activate
8. Instalar as bibliotecas com o comando: pip install -r requirements.txt
9. Rodar o arquivo com o comando: python main.py

## Etapas de desenvolvimento

### 1. Definição do formato

O projeto foi primeiro desenvolvido em um arquivo Python, para que o código possa ser guardado no Github mais facilmente. Após todo o código ser escrito, um notebook foi criado a partir desse código para mais fácil compartilhamento e visualização do projeto.

### 2. Importação e compreensão dos dados

Para deixar a importação dos arquivos ainda mais conveniente, o notebook utiliza a biblioteca do Kagglehub para baixar o dataset para cache. Para a versão em código, o arquivo CSV do dataset deve ser baixado manualmente e colocado na mesma pasta que o arquivo main.py. O CSV não está incluído no repositório github, mas será enviado na entrega do projeto pelo AVA.

Um arquivo com duas funções utilitárias criadas ao longo do desenvolvimento do projeto também foi produzido. Essas funções estão presentes em uma célula separada na versão notebook.

Após a importação, foram executados os métodos head e info para a checagem inicial do dataset. Todas as colunas possuem tipos adequados, então nenhuma conversão foi necessária

### 3. Tratamento e preparação dos dados

Inicialmente foi feita a identificação de duplicados, em que 26.80% do dataset foi identificado como "duplicado". Porém, decidi não remover esses supostos duplicados por duas razões: primeiro que não há ID do cliente ou da reserva, então não dá para saber se realmente são reservas duplicadas ou se são reservas distintas com os mesmos parâmetros; e também não quis remover mais de 1/4 do dataset sem ter certeza absoluta que esses duplicados eram reais.

Após isso foi feito o tratamento de nulos. A coluna company é 94.31 nula, o que a tornou uma candidata para ser removida por completo, mas ao invés disso decidi preencher esses nulos com uma nova categoria. Os nulos de agent são apenas 13.69%, mas decidi manter (substituindo por uma nova categoria) pois eles podem ser interpretados como reservas sem agência o que é uma opção válida. Já os nulos de country e children foram removidos pois sua falta não faz sentido.

As features foram distribuídas entre numéricas e categóricas já nessa etapa para facilitar a geração de box plots. À primeira vista somente o total_of_special_requests me pareceu um bom candidato para a remoção de outliers, então fiz também a identificação de outliers pelo método do IQR. Decidi não remover nenhum outlier nessa etapa, pois não sabia se haviam relações importantes para a modelagem preditiva.

### 4. Análise exploratória

Primeiramente foi gerada uma matriz de correlação entre todas as features numéricas no dataset. Foi possível identificar duas correlações interessantes:

1. stays_in_weekend_nights com stays_in_week_nights (estadia total)
2. children e adults com adr (hotéis cobram por hóspede).

Após isso foram feitos gráficos de colunas relevantes no dataset. Em geral, foram gerados dois gráficos para cada coluna: um mostrando a proporção de cada categoria/distribuição da variável, e um relacionando a coluna com cancelamentos.

Nessa etapa foram identificados e removidos outliers das colunas de lead_time e adr, o que facilitou bastante a visualização do seu comportamento. As colunas de previous_cancellations e booking_changes também se mostraram boas candidatas para a remoção de outliers, mas resolvi não fazer isso pois elas continham muito menos valores possíveis que lead_time e adr e achei que isso não iria interferir muito no modelo.

### 5. Principais Insights

A partir da análise exploratória, foi possível extrair os seguintes insights:

1. Há uma correlação entre stays_in_weekend_nights com stays_in_week_nights. Isso parece meio óbvio à primeira vista mas pode indicar que os hotéis fazem apenas (ou em maioria) reservas com dias seguidos e não intermitentes.
2. Há uma correlação entre children e adr e adults e adr. Indica que os hotéis cobram por número de hóspedes.
3. A maior parte dos cancelamentos é no mesmo dia, o que me surpreendeu um pouco.
4. O número de cancelamentos tende a diminuir com o aumento de lead_time, mas não é uma curva "limpa".
5. Os meses com mais reservas e cancelamentos são Julho e Agosto.
6. A distribuição do ADR tem um pico em 62 e após isso tende a baixar. A taxa de cancelamentos entre ADR 0 e 62 é relativamente baixa, o que é interessante.
7. Clientes repetidos tendem a cancelar menos, e clientes que já cancelaram tendem a cancelar novamente.
8. Clientes com mudanças na reserva e pedidos especiais tendem a cancelar menos.

### 6. Modelagem Preditiva

Para a modelagem preditiva decidi fazer mais duas features: uma com o total de dias da estadia, e outra com o total de hóspedes na reserva (o que apareceu em uma das correlações). O ano da reserva foi removido das features pois não achei que era um bom valor para fazer a previsão, já que as reservas ao longo de um ano dependem de vários fatores externos como a economia, eventos mundiais impactando a região do hotel, etc. Também removi o dia do mês da reserva pois não achei uma informação útil. Removi o reservation_status junto com o target (is_canceled) pois ele também indica os cancelamentos.

Após separar as features e o target, foi feito um novo dataframe apenas para ter uma melhor visualização e validação dos dados escolhidos.

Os dados de treino e teste foram separados em uma proporção de 80% para treino. Para as features numéricas foi usado o StandardScaler, e para as categoricas foi usado o OneHotEncoder. Foram geradas mais de 900 colunas adicionais, e talvez algum outro encoder para as features categoricas teria sido melhor.

Treinei dois modelos sobre esses dados: um de Regressão Logística e um de Random Forest. Em geral, o modelo da Regressão Logística apresentou uma maior accuracy e AUC. Ao observar as matrizes de confusão, foi possível notar que ambos modelos tiveram uma certa dificuldade prevendo cancelamentos, o Random Forest em particular teve a maioria de seus erros como falsos negativos, indicado pelo péssimo recall de 39.69%. As curvas ROC de ambos os modelos estão bem parecidas, e ambas longe do ideal.

Em conclusão, mesmo que o modelo de Regressão Logística tenha se saído melhor que o Random Forest, eu não sei se usaria ele em uma situação real devido ao seu recall menor que 75%. Talvez a remoção do restante dos outliers candidatos melhore ou modelo, ou talvez haja pouca correlação entre os dados.
