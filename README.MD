# Projeto: Coleta e Processamento de Dados do Brasileirão

Este projeto tem como objetivo extrair dados dos jogos do Campeonato Brasileiro (como times, placares, data, hora, estádio, cidade, rodada) e integrá-los com informações climáticas, para, futuramente, desenvolver modelos de previsão (por exemplo, previsão de vitória do mandante).


# Métodos de Coleta de Dados
Melhor Coleta dos Dados:
Para obter os dados do campeonato brasileiro de forma mais precisa, recomenda-se o uso de scripts de automatização (por exemplo, com PyAutoGUI) para acessar diretamente:
https://portaldegovernanca.cbf.com.br/documentos-da-partida

Web Scraping
Descrição: Utiliza bibliotecas como requests e BeautifulSoup (ou Selenium para casos mais dinâmicos) para acessar a página e extrair informações.

Pontos de Atenção:

As páginas podem ser atualizadas, o que exige manutenção constante do código.

Erros podem ocorrer se os elementos da página forem alterados.

Automatização de Tela
Descrição: Utiliza PyAutoGUI para simular cliques, capturar screenshots da tela e, em seguida, extrair o texto utilizando OCR (como EasyOCR ou Tesseract).

Fluxo:

Automatização e captura de tela.

Extração de texto com EasyOCR (ou Tesseract).

Armazenamento inicial dos dados em CSV.

Pós-processamento (com OpenCV) para ordenação e limpeza.

Normalização de nomes de cidades e correção de dados.

Geração do arquivo final cidades_partidas_corrigido.csv.

Problemas Encontrados e Soluções
API da CBF e Certificados SSL
Problema:
Ao usar a API da CBF, ocorreram muitos erros relacionados a certificados SSL.

Causa:
O servidor da CBF usa um certificado que não é reconhecido como válido por algumas bibliotecas Python, inclusive no Colab.

Observação:
Esse erro é do lado do servidor da CBF e pode ser difícil de contornar.

Manutenção do Código (Web Scraping)
Problema:
As páginas podem modificar seus elementos ou estrutura, exigindo manutenção constante do código de webscraping.

Solução:
Monitorar possíveis alterações na estrutura da página e atualizar as expressões regulares ou seletores conforme necessário.

Desafios com OCR
Problema:
Ferramentas de OCR (ex.: EasyOCR) enfrentaram dificuldades com caracteres especiais e acentuação.

Soluções Adotadas:

Forçar o idioma: Utilizar lang='por' para melhorar o reconhecimento dos caracteres acentuados.

Regex Tolerante: Ajustar as expressões regulares para capturar nomes de cidades mesmo com variações.

Normalização e Correção: Aplicar normalização de caracteres (ex.: remover acentos indevidos).

Problema Adicional:
Em alguns casos, o OCR pode interpretar valores incorretamente (ex.: média de cartão amarelado ou vermelho sendo lida como horário).

Solução:

Filtrar valores de hora: Considerar somente horários entre 06:00 e 23:59.

Ignorar valores que não sejam horários válidos (ex.: 0.27).

Substituir ponto por dois-pontos (ex.: transformar "21.00" em "21:00").

Visual Crossing API – Coleta de Clima
Problema:
A API da Visual Crossing, ao usar o parâmetro include=hours, retorna 24 registros por chamada (um para cada hora do dia), mesmo que você precise de dados para apenas uma hora.

Consequência:
Isso pode levar à mensagem de erro:

"You have exceeded the maximum number of daily result records for your account."

Solução:
Agrupar chamadas por data ou filtrar localmente os dados para extrair somente a hora desejada, evitando chamar a API desnecessariamente.

Pipeline do Código
O fluxo de execução do código é o seguinte:

Automatização de Tela e Captura de Imagem:
Utiliza PyAutoGUI para navegar pelo site e capturar screenshots dos dados.

Extração de Texto e OCR:

EasyOCR: Utilizado para extrair texto dos screenshots.

Se os resultados não forem satisfatórios, é sugerido utilizar Tesseract (lembre de definir o caminho para o executável):

python
Copiar
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Armazenamento Inicial:
Os dados extraídos são salvos em um CSV.

Pós-processamento:

Ordenação dos jogos por rodada.

Limpeza e normalização dos nomes das cidades (removendo ", Brazil" etc).

Correção de nomes e formatação.

Geração do CSV Final:
A saída final é armazenada em cidades_partidas_corrigido.csv.

Integração com Dados Climáticos:
Utiliza-se a API do Open-Meteo para obter informações de clima (temperatura, umidade, vento) para cada jogo com base na cidade e data/hora.

Problemas na Leitura dos Dados
Qualidade das Imagens:
Imagens com baixa resolução, compressão (JPEG) ou contraste inadequado prejudicam a performance do OCR.

Solução:
Utilizar ferramentas como UpScayl para melhorar a qualidade das imagens.
O UpScayl pode ser baixado em: https://upscayl.org

Pytesseract:
Apesar do UpScayl ajudar, em alguns casos, o Tesseract (usado via pytesseract) pode ser necessário.

Lembre-se de definir o caminho para o executável do Tesseract, conforme mostrado acima.

Mesmo imagens aparentemente idênticas podem apresentar diferenças sutis que impactam a acurácia do OCR (devido à compressão, resolução ou artefatos visuais).

Considerações Finais
O projeto envolve diversas abordagens (webscraping, automação de tela, OCR) para extrair e processar dados complexos do Brasileirão. Cada método possui seus desafios e requer manutenção e ajustes constantes devido às variações nos dados e na forma como os sites apresentam as informações.