caso faça webscrapping:
pip install requests beautifulsoup4 pandas
pip install selenium pandas webdriver-manager

usando por cliques:
pip install pyautogui pyperclip pandas
pip install pyperclip

Se você usar a API da cbf vai ter muitos erros, pois:
O site da CBF está usando um certificado SSL que não está sendo reconhecido como válido por várias bibliotecas Python, inclusive no Colab (que já vem com tudo configurado certinho). Isso é erro do servidor da CBF — provavelmente

Problemas de usar webscrapping
as vezes as páginas mudou os elementos e aí tem que ficar de olho com manutenção do código e possíveis mudanças

Problema?
OCR tem problema com caracteres especiais, Tesseract às vezes interpreta errado sem a linguagem correta

Solução: 
Forçar lang='por' para melhorar reconhecimento de acentos.
Regex mais tolerante para pegar cidade mesmo que tenha variações.
Correção pós-OCR com normalização de caracteres.

Problema?
as vezes pega média de cartão amarelho ou vermelho em vez de horário

Solução
Filtro de hora: só considera horas entre 06:00 e 23:59.
Ignora valores como 0.27, que não são horários válidos.
Substitui . por : para lidar com formatos como 21.00


Problema da visual crossing (api do clima)
Ou seja, vira 380 x 24
Cada chamada include=hours = até 24 registros
Mesmo se você quer apenas 1 hora, ele retorna o dia inteiro de horas por padrão
aí vai aparecer uma mensagem como: You have exceeded the maximum number of daily result records for your account. 


Lógica do código:
Automatização de Tela com PyAutoGUI + Screenshot
Extração de Texto com EasyOCR + Armazenamento CSV Inicial
Pós-processamento das imagens OCR (via OpenCV)
Ordenação por Rodada
Limpeza e Normalização de Cidades (remover , Brazil)
Correção de Nomes de Cidades
Saída Final no CSV cidades_partidas_corrigido.csv


ERROS NA LEITURA DOS DADOS:
Problema grande que foi a resolução da qualidade das imagens, para
identificar times e placares, ferramentas como easyocr não funcionou, para contornar isso utilizei o upscayl. Para renderizar tudo de uma vez sem ser manualmente, escolha pasta que tem todos os prints 
Você pode baixar o upscayl por aqui:
https://upscayl.org

Apesar do upscayl melhorar a qualidade não foi suficiente para todos os jogos ainda sim, apresentava inconsistência então para resolver de forma definitiva é necessário que baixe o baixe o pytesseract
lembre de Define o caminho para o executável do tesseract

import pytesseract
# Define o caminho para o executável do tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

mesmo com imagens visualmente idênticas, pequenas diferenças de compressão, resolução, contraste, ou artefatos de compressão (como JPEG) podem atrapalhar bastante o OCR do Tesseract