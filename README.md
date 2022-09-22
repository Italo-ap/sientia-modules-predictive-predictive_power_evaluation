[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ihmstefanini/pandorabox/streamlitcloudversion/pandora_app.py)

# Welcom to Sientia Predictive Power Evaluator 👋

```
** The fastest way to assess the predictive potential of a data problem in the industry**

Pandora Box enables analysts and data scientists to discover the predictive potential of an industrial process in minutes.

Pandora does quick feasibility analysis of the problems you want to predictively model in the industry (quality prediction, virtual sensors, etc).
```

O Pandora Box possibilita que analistas e cientistas de dados descubram o potencial preditivo de um processo industrial em minutos.

O Pandora faz análises rápidas de viabilidade dos problemas que deseja modelar preditivamente na indústria (predição de qualidade, sensores virtuais, etc).

![IHM Pandora](/images/IHM%20Pandora02.png)

O Pandora pode ser usado para analisar correlações entre variáveis de processo e qualidade (correlação linear e não linear), fazer teste de hipóteses (verificar causalidade), calcular o tempo de atraso, sincronizar as variáveis no tempo e fornecer informações sobre o potencial preditivo do problema.

### Por quê o Pandora?

Sabe aqueles momentos em que você precisa fazer uma rápida análise de viabilidade (potencial preditivo) de um problema e não tem tempo para fazer todo aquela codificação em python ou não tem ninguém disponível pra te ajudar?

Se você já esteve nessa situação, então o Pandora é o aplicativo perfeito para te salvar nesses momentos!

### Entendendo o Pandora

Analisar se uma variável de qualidade de um processo industrial pode ser modelada preditivamente é uma tarefa complexa e que exige avaliar uma séria de informações. O Pandora facilita esse processo, fazendo todas as etapas necessárias para isso de maneira mais automática e rápida.

Claro que não reinventamos a roda. Entretanto, tornamos esse processo mais democrático e acessível para os analistas, engenheiros e tomadores de decisão na indústria, permitindo que uma análise mínima possa ser feita sobre os dados que se deseja explorar/modelar.

Tal análise preliminar, utilizando o Pandora Box, permite que o usuário tenha um mínimo de informação necessária a respeito do potencial preditivo do problema, facilitando a tomada de decisão sobre seguir ou não com o projeto de dados.

Pandora também disponibiliza para o usuário um dataset sincronizado e pronto para ser explorado de forma mais consistente e sistemática.

### Instalação

1. Clone o projeto e navegue até a pasta project

```
git clone https://github.com/ihmstefanini/pandorabox.git
cd pandorabox
```

2. Rode seu ambiente virtual (nesse exemplo iremos utilizar o virtualenv)

```
python -m venv myenv
```

Linux

```
source myenv/bin/activate
```

Windows

```
. myenv/Scripts/activate
```

3. Instale as dependencias

```
pip install -r requirements.txt
```

4. Rode a aplicação no terminal de comando

```
streamlit run pandora_app.py
```

### Utilizando o Pandora

- Um exemplo prático

### Credenciais de acesso

- **User Name:** default
- **Password:** 12345

**Carregando os dados no Pandora**

![IHM Pandora01-demo01](/images/Pandora-Gif01_a.gif)

**Visualizando os dados e suas características básicas**

![IHM Pandora01-demo02](/images/Pandora-Gif02.gif)

**Visualizando e interpretando as correlações**

![IHM Pandora01-demo03](/images/Pandora-Gif03.gif)

### Junte-se a nossa comunidade de analistas que utilizam o Pandora

- Entre no chat do [Teams](https://teams.microsoft.com/l/team/19%3aac1d8e5b18d74945a252fae738c6c0e5%40thread.tacv2/conversations?groupId=36d4af41-3c0a-41ad-9e71-8bcb8bdc4c7d&tenantId=d8bde65a-3ded-4346-9518-670204e6e184)

### Reportando bugs e contribuindo com o código

Quer reportar um bug ou solicitar uma feature? Fale com a gente no [Teams](https://teams.microsoft.com/l/team/19%3aac1d8e5b18d74945a252fae738c6c0e5%40thread.tacv2/conversations?groupId=36d4af41-3c0a-41ad-9e71-8bcb8bdc4c7d&tenantId=d8bde65a-3ded-4346-9518-670204e6e184)) ou abra um issue.

Quer ajudar a construir o Pandora? Em breve!

### Suporte ou Contato

Mande um email pra gente no inteligenciaindustrial@ihm.com.br.

### Pandorabox for teams e provisionado em cloud

Quer compartilhar suas descobertas, exportar os datasets e avaliar o potencial preditivo de um problema em larga escala?

Entre na nossa [**lista de espera do Pandorabox Cloud**](https://airtable.com/shrXwO3hOV5KK9MGH)

### Licença

Nosso aplicativo é open source sobre os termos de uso da licença GPL 3.0
