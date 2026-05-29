# Desafio Extra - Hotel Booking Demand

Desafio extra do curso "Introdução à Inteligência Artificial" do programa [SCTEC](https://sctec.scti.sc.gov.br/).

Elaborado por [Julio Gonçalves Ramos](https://www.linkedin.com/in/julio-ramos-1684a5390/).

Link para o repositório: [https://github.com/juliogramos/sctec-passos-ia-final](https://github.com/juliogramos/sctec-passos-ia-final)

Link para o Notebook: TBD

## Tecnologias

- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## Como visualizar localmente

### Executar Notebook

1. Acessar o notebook: TBD
2. Executar células em ordem OU usar o botão Run All
   Não é preciso baixar o dataset manualmente, ele é baixado através da biblioteca do Kagglehub e salvo em cache.

### Executar arquivo Python

1. Clonar o repositório ou baixar os arquivos e extrair em uma pasta
2. Baixar o dataset: [link](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
3. Colocar o arquivo CSV na mesma pasta que o arquivo main.py e utilitarios.py
4. Instalar o Python (versão utilizada: 3.12.3) e o Pip
5. Instalar a biblioteca virtualenv com o comando: pip3 install virtualenv
6. Criar um novo ambiente virtual com o comando: virtualenv venv
7. Ativar o ambiente virtual com o comando (Mac e Linux): source venv/bin/activate
8. Instalar as bibliotecas com o comando: pip install -r requirements.txt
9. Rodar o arquivo com o comando: python main.py

Algumas dos arquivos de imagem gerados ficaram com a legenda cortada, então recomendo ver os gráficos rodando o programa.

## Etapas de desenvolvimento

### 1. Definição do formato

O projeto foi primeiro desenvolvido em um arquivo Python, para que o código possa ser guardado no Github mais facilmente. Após todo o código ser escrito, um notebook foi criado a partir desse código para mais fácil compartilhamento e visualização do projeto.

### 2. Importação e compreensão dos dados

### 3. Tratamento e preparação dos dados

### 4. Análise exploratória

### 5. Principais Insights

1. Há uma correlação entre stays_in_weekend_nights com stays_in_week_nights. Isso parece meio óbvio à primeira vista mas indica que os hotéis fazem apenas (ou em maioria) reservas com dias seguidos e não intermitentes.
2. Há uma correlação entre children e adr e adults e adr. Indica que os hotéis cobram por número de hóspedes.
3. A maior parte dos cancelamentos é no mesmo dia, o que me surpreendeu um pouco.
4. O número de cancelamentos tende a diminuir com o aumento de lead_time, mas não é uma curva "limpa".
5. Os meses com mais reservas e cancelamentos são Julho e Agosto.
6. O ADR tem um pico em 62 e após isso tende a baixar. A taxa de cancelamentos entre ADR 0 e 62 é relativamente baixa, o que é interessante.
7. Clientes repetidos tendem a cancelar menos, e clientes que já cancelaram tendem a cancelar novamente.
8. Clientes com mudanças na reserva e pedidos especiais tendem a cancelar menos.

### 6. Modelagem Preditiva
