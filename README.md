Como funciona

1. Você seleciona a região desejada com o mouse (ou informa as coordenadas).
2. O programa renderiza só essa região em alta resolução.
3. Se houver texto "real" na região, usa-o; senão, aplica OCR (Tesseract).
4. As palavras são reagrupadas em linhas e colunas -> tabela.
5. Salva em Excel (.xlsx), CSV ou TXT.

A região é guardada em frações da página (0 a 1), então funciona igual em
qualquer resolução e pode ser reutilizada em vários PDFs de mesmo layout.

Instalação

    pip install pymupdf pytesseract pandas openpyxl pillow

    + Tesseract OCR (programa externo):
      Windows: https://github.com/UB-Mannheim/tesseract/wiki  (marque "Portuguese")
      Ubuntu : sudo apt install tesseract-ocr tesseract-ocr-por
      macOS  : brew install tesseract tesseract-lang

Exemplos de uso

    # 1) Abre janela para desenhar a região, extrai e salva a região em regiao.json
    python extrator_pdf.py relatorio.pdf --salvar-config regiao.json

    # 1b) Vários PDFs: a janela de seleção abre para CADA PDF (Esc pula um arquivo)
    python extrator_pdf.py C:\\relatorios --recursivo -o consolidado.xlsx

    # 1c) Vários PDFs, mas desenhando a região uma única vez para todos
    python extrator_pdf.py C:\\relatorios --regiao-unica -o consolidado.xlsx

    # 2) Reutiliza a região em vários PDFs (todas as páginas), gerando um só Excel
    python extrator_pdf.py *.pdf --config regiao.json -o consolidado.xlsx

    # 2b) Uma pasta inteira (e subpastas), tudo em um único Excel
    python extrator_pdf.py C:\\relatorios --recursivo --config regiao.json -o consolidado.xlsx

    # 3) Sem interface gráfica: região em frações (x0,y0,x1,y1) da página
    python extrator_pdf.py relatorio.pdf --regiao 0.05,0.30,0.95,0.70 --paginas 2-4

    # 4) Converter "1.234,56" em número e usar a 1ª linha como cabeçalho
    python extrator_pdf.py relatorio.pdf --config regiao.json --numeros --cabecalho
