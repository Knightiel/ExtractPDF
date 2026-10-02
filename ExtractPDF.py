from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import statistics
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps


# Manual Configuration PATH TESSERACT (opcional)

TESSERACT_CAMINHO = r""

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # versões antigas
    import fitz



# Estruturas básicas

@dataclass
class Palavra:
    texto: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def altura(self) -> float:
        return self.y1 - self.y0



# PDF -> imagem / texto nativo

def retangulo_da_regiao(pagina, regiao):
    """Converte a região em frações (x0,y0,x1,y1) para um retângulo em pontos PDF."""
    r = pagina.rect
    fx0, fy0, fx1, fy1 = regiao
    return fitz.Rect(
        r.x0 + fx0 * r.width,
        r.y0 + fy0 * r.height,
        r.x0 + fx1 * r.width,
        r.y0 + fy1 * r.height,
    )


def renderizar(pagina, retangulo=None, dpi=300) -> Image.Image:
    """Renderiza a página (ou só um retângulo dela) como imagem PIL."""
    pix = pagina.get_pixmap(dpi=dpi, clip=retangulo, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def palavras_nativas(pagina, retangulo) -> list[Palavra]:
    """Palavras do texto embutido no PDF (não funciona para imagens)."""
    brutas = pagina.get_text("words", clip=retangulo)
    return [Palavra(w[4], w[0], w[1], w[2], w[3]) for w in brutas if w[4].strip()]



# Pré-processamento para OCR

def limiar_otsu(cinza: np.ndarray) -> int:
    """Calcula o limiar de binarização de Otsu."""
    hist = np.bincount(cinza.ravel(), minlength=256).astype(float)
    total = cinza.size
    soma_total = float(np.dot(np.arange(256), hist))
    peso_b = soma_b = melhor = 0.0
    limiar = 127
    for t in range(256):
        peso_b += hist[t]
        if peso_b == 0:
            continue
        peso_f = total - peso_b
        if peso_f == 0:
            break
        soma_b += t * hist[t]
        var = peso_b * peso_f * (soma_b / peso_b - (soma_total - soma_b) / peso_f) ** 2
        if var > melhor:
            melhor, limiar = var, t
    return limiar


def preprocessar(img: Image.Image, remover_linhas=True) -> Image.Image:
    """Tons de cinza -> binarização -> remoção das linhas da tabela -> margem."""
    cinza = ImageOps.autocontrast(ImageOps.grayscale(img))
    arr = np.array(cinza)
    escuro = arr <= limiar_otsu(arr)  # True = tinta

    if remover_linhas:
        # Linhas de grade longas confundem o Tesseract: apaga as que cruzam
        # mais da metade da largura/altura do recorte.
        h_mask = escuro.mean(axis=1) > 0.5
        v_mask = escuro.mean(axis=0) > 0.5
        for eixo_mask, eixo in ((h_mask, 0), (v_mask, 1)):
            idx = np.where(eixo_mask)[0]
            for i in idx:  # apaga também 1 px de cada lado (antialiasing)
                for j in (i - 1, i, i + 1):
                    if 0 <= j < escuro.shape[eixo]:
                        if eixo == 0:
                            escuro[j, :] = False
                        else:
                            escuro[:, j] = False

    saida = Image.fromarray(np.where(escuro, 0, 255).astype(np.uint8))
    return ImageOps.expand(saida, border=20, fill=255)



# OCR

def configurar_tesseract(cmd: str | None, idiomas: str) -> str:
    """Valida o Tesseract e devolve os idiomas realmente instalados."""
    cmd = cmd or TESSERACT_CAMINHO or None
    if not cmd and not shutil.which("tesseract"):
        # Não está no PATH: procura nos locais de instalação padrão do Windows
        for candidato in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if Path(candidato).is_file():
                cmd = candidato
                break
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception as erro:
        usado = pytesseract.pytesseract.tesseract_cmd
        sys.exit(
            "ERRO: não consegui executar o Tesseract OCR.\n"
            f"  Caminho tentado : {usado}\n"
            f"  Detalhe do erro : {type(erro).__name__}: {erro}\n\n"
            "O que fazer:\n"
            "  1) Descubra onde o tesseract.exe foi instalado (procure em C:\\Program Files).\n"
            "  2) Rode novamente informando o caminho completo, por exemplo:\n"
            '     --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n'
            "  3) Se o caminho estiver certo, teste no terminal:  tesseract --version"
        )
    disponiveis = set(pytesseract.get_languages(config=""))
    pedidos = [l for l in idiomas.split("+") if l]
    ok = [l for l in pedidos if l in disponiveis]
    faltando = [l for l in pedidos if l not in disponiveis]
    if faltando:
        print(f"AVISO: idioma(s) do Tesseract não instalado(s): {', '.join(faltando)}",
              file=sys.stderr)
    if not ok:
        print("AVISO: usando 'eng' como alternativa.", file=sys.stderr)
        ok = ["eng"]
    return "+".join(ok)


def palavras_ocr(img: Image.Image, idioma: str, psm: int, conf_min: float) -> list[Palavra]:
    d = pytesseract.image_to_data(
        img, lang=idioma, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    palavras = []
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        if not txt or float(d["conf"][i]) < conf_min:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        palavras.append(Palavra(txt, x, y, x + w, y + h))
    return palavras



# Palavras -> linhas -> colunas -> tabela

def agrupar_linhas(palavras: list[Palavra], tolerancia=0.6) -> list[list[Palavra]]:

    if not palavras:
        return []
    alt = statistics.median(p.altura for p in palavras)
    linhas: list[list[Palavra]] = []
    for p in sorted(palavras, key=lambda p: p.cy):
        if linhas and abs(p.cy - statistics.mean(q.cy for q in linhas[-1])) <= tolerancia * alt:
            linhas[-1].append(p)
        else:
            linhas.append([p])
    return linhas


def detectar_separadores(palavras, n_linhas, fator_gap=0.7, tolerancia=0.0) -> list[float]:

    alt = statistics.median(p.altura for p in palavras)
    celula = max(alt / 4, 1e-6)
    x_min = min(p.x0 for p in palavras)
    x_max = max(p.x1 for p in palavras)
    n = int((x_max - x_min) / celula) + 2
    cont = np.zeros(n, dtype=int)
    for p in palavras:
        cont[int((p.x0 - x_min) / celula): int((p.x1 - x_min) / celula) + 1] += 1

    vazio = cont <= int(tolerancia * n_linhas)
    seps, i = [], 0
    while i < n:
        if vazio[i]:
            j = i
            while j < n and vazio[j]:
                j += 1
            if i > 0 and j < n and (j - i) * celula >= fator_gap * alt:
                seps.append(x_min + (i + j) / 2 * celula)
            i = j
        else:
            i += 1
    return seps


def montar_tabela(palavras, fator_gap=0.7, tolerancia=0.0) -> list[list[str]]:
    linhas = agrupar_linhas(palavras)
    if not linhas:
        return []
    seps = detectar_separadores(palavras, len(linhas), fator_gap, tolerancia)
    tabela = []
    for linha in linhas:
        celulas = [[] for _ in range(len(seps) + 1)]
        for p in sorted(linha, key=lambda p: p.x0):
            celulas[bisect_right(seps, p.cx)].append(p.texto)
        tabela.append([" ".join(c) for c in celulas])
    return tabela



# Pós-processamento

def para_numero(valor: str):
    """'1.234,56' -> 1234.56 | '12%' -> 12.0 | mantém texto que não for número."""
    t = valor.strip().replace("R$", "").replace("%", "").replace(" ", "")
    if not t:
        return valor
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?|[-+]?\d+(,\d+)?", t):
        if "," not in t and len(t.lstrip("+-")) > 1 and t.lstrip("+-").startswith("0"):
            return valor  # códigos como 00123 preservam os zeros
        t = t.replace(".", "").replace(",", ".")
        return float(t) if "." in t else int(t)
    return valor



# Seleção da região com o mouse (Tkinter)

def selecionar_regiao_gui(caminho_pdf: str, pagina_inicial=0, rotulo=""):
    """
    Abre uma janela para desenhar o retângulo.
    Retorna (pagina, (x0,y0,x1,y1)); None se o usuário pressionar Esc (pular este PDF).
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk
    except ImportError:
        sys.exit("Tkinter indisponível. Use --regiao x0,y0,x1,y1 (frações de 0 a 1).")

    doc = fitz.open(caminho_pdf)
    root = tk.Tk()
    root.title(f"{rotulo or Path(caminho_pdf).name} — arraste para selecionar | "
               "←/→ troca página | Enter confirma | Esc pula este PDF")
    max_w, max_h = root.winfo_screenwidth() * 0.85, root.winfo_screenheight() * 0.80

    est = {"pag": min(pagina_inicial, len(doc) - 1), "ini": None, "ret": None,
           "id": None, "dim": (1, 1), "foto": None, "resultado": None, "pular": False}
    info = tk.Label(root, anchor="w")
    canvas = tk.Canvas(root, cursor="cross", highlightthickness=0)
    info.pack(fill="x")
    canvas.pack()

    def mostrar():
        img = renderizar(doc[est["pag"]], dpi=110)
        esc = min(max_w / img.width, max_h / img.height, 1.0)
        img = img.resize((int(img.width * esc), int(img.height * esc)))
        est.update(foto=ImageTk.PhotoImage(img), dim=img.size, ret=None, id=None)
        canvas.config(width=img.width, height=img.height)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=est["foto"])
        info.config(text=f"{rotulo or Path(caminho_pdf).name}  |  Página {est['pag'] + 1}/"
                         f"{len(doc)}  |  desenhe um retângulo sobre a tabela e pressione "
                         "Enter  (Esc = pular este PDF)")

    def pressionar(e):
        est["ini"] = (e.x, e.y)
        if est["id"]:
            canvas.delete(est["id"])
        est["id"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

    def arrastar(e):
        x0, y0 = est["ini"]
        canvas.coords(est["id"], x0, y0, e.x, e.y)

    def soltar(e):
        x0, y0 = est["ini"]
        w, h = est["dim"]
        xa, xb = sorted((max(0, min(x0, w)), max(0, min(e.x, w))))
        ya, yb = sorted((max(0, min(y0, h)), max(0, min(e.y, h))))
        est["ret"] = (xa / w, ya / h, xb / w, yb / h) if xb - xa > 5 and yb - ya > 5 else None

    def mudar(delta):
        nova = est["pag"] + delta
        if 0 <= nova < len(doc):
            est["pag"] = nova
            mostrar()

    def confirmar(_=None):
        if not est["ret"]:
            messagebox.showwarning("Região", "Desenhe um retângulo antes de confirmar.")
            return
        est["resultado"] = (est["pag"], est["ret"])
        root.destroy()

    canvas.bind("<ButtonPress-1>", pressionar)
    canvas.bind("<B1-Motion>", arrastar)
    canvas.bind("<ButtonRelease-1>", soltar)
    root.bind("<Left>", lambda e: mudar(-1))
    root.bind("<Right>", lambda e: mudar(1))
    root.bind("<Return>", confirmar)
    def pular(_=None):
        est["pular"] = True
        root.destroy()

    root.bind("<Escape>", pular)
    mostrar()
    root.mainloop()

    if est["resultado"]:
        return est["resultado"]
    if est["pular"]:
        return None
    sys.exit("Seleção cancelada.")



# Extração de uma página

def extrair_pagina(pagina, regiao, args, idioma) -> tuple[list[list[str]], Image.Image | None]:
    ret = retangulo_da_regiao(pagina, regiao)
    palavras, recorte = [], None

    if args.modo in ("auto", "nativo"):
        palavras = palavras_nativas(pagina, ret)
        if args.modo == "auto" and len(palavras) < 3:
            palavras = []  # quase sem texto embutido => provavelmente imagem

    if not palavras and args.modo != "nativo":
        recorte = renderizar(pagina, ret, dpi=args.dpi)
        preparada = preprocessar(recorte, remover_linhas=not args.manter_linhas)
        palavras = palavras_ocr(preparada, idioma, args.psm, args.conf_min)

    if not palavras:
        return [], recorte
    if args.texto:
        return [[" ".join(p.texto for p in sorted(linha, key=lambda p: p.x0))]
                for linha in agrupar_linhas(palavras)], recorte
    return montar_tabela(palavras, args.gap, args.tolerancia), recorte


def interpretar_paginas(texto: str | None, total: int, padrao: list[int]) -> list[int]:
    """'1,3-5' -> [0,2,3,4] (índices base 0). 'todas' -> todas."""
    if texto is None:
        return [p for p in padrao if p < total]
    if texto.strip().lower() in ("todas", "all", "*"):
        return list(range(total))
    paginas = set()
    for parte in texto.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-")
            paginas.update(range(int(a) - 1, int(b)))
        elif parte:
            paginas.add(int(parte) - 1)
    return sorted(p for p in paginas if 0 <= p < total)



# Escolha dos PDFs quando nenhum é informado na linha de comando

def escolher_pdfs() -> tuple[list[str], bool]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        recursivo = False
        arquivos = filedialog.askopenfilenames(
            title="Selecione um ou mais PDFs (Ctrl/Shift para vários) — "
                  "Cancelar = escolher uma PASTA inteira",
            filetypes=[("PDF", "*.pdf")])
        if not arquivos:
            pasta = filedialog.askdirectory(title="Selecione a pasta com os PDFs")
            arquivos = [pasta] if pasta else []
            if pasta:
                recursivo = messagebox.askyesno(
                    "Subpastas", "Incluir também os PDFs das subpastas?")
        root.destroy()
        return list(arquivos), recursivo
    except Exception:
        entrada = input("Caminho do PDF ou da pasta (separe vários por ';'): ").strip().strip('"')
        itens = [c.strip().strip('"') for c in entrada.split(";") if c.strip()]
        recursivo = False
        if any(Path(i).is_dir() for i in itens):
            recursivo = input("Incluir subpastas? (s/n): ").strip().lower().startswith("s")
        return itens, recursivo


def expandir_entradas(entradas: list[str], recursivo=False) -> list[str]:
    achados: list[Path] = []
    for item in entradas:
        caminho = Path(item)
        if caminho.is_dir():
            busca = caminho.rglob if recursivo else caminho.glob
            achados += [f for f in busca("*") if f.suffix.lower() == ".pdf"]
        elif any(ch in item for ch in "*?["):
            achados += [Path(f) for f in glob.glob(item, recursive=recursivo)
                        if f.lower().endswith(".pdf")]
        else:
            achados.append(caminho)
    vistos, unicos = set(), []
    for f in sorted(achados, key=lambda x: str(x).lower()):  # ordem alfabética, sem repetir
        if str(f.resolve()) not in vistos:
            vistos.add(str(f.resolve()))
            unicos.append(str(f))
    return unicos


def escolher_saida(padrao: Path) -> Path:
    """No modo interativo, pergunta onde salvar o arquivo final."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        escolhido = filedialog.asksaveasfilename(
            title="Salvar resultado como", initialfile=padrao.name,
            initialdir=str(padrao.parent), defaultextension=padrao.suffix,
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("Texto", "*.txt")])
        root.destroy()
        return Path(escolhido) if escolhido else padrao
    except Exception:
        return padrao



# Main

def criar_parser():
    p = argparse.ArgumentParser(
        description="Extrai uma região/tabela (inclusive em imagem) de PDFs usando OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exemplos de uso")[1] if "Exemplos de uso" in __doc__ else "",
    )
    p.add_argument("pdfs", nargs="*",
                   help="um ou mais arquivos PDF (se omitido, abre uma janela para escolher)")
    p.add_argument("-o", "--saida", help="arquivo de saída (.xlsx, .csv ou .txt)")
    p.add_argument("--recursivo", action="store_true",
                   help="ao receber uma pasta, procura PDFs também nas subpastas")

    g = p.add_argument_group("região")
    g.add_argument("--regiao", help="x0,y0,x1,y1 em frações da página (0 a 1)")
    g.add_argument("--config", help="carrega a região de um arquivo JSON")
    g.add_argument("--salvar-config", help="salva a(s) região(ões) escolhida(s) em JSON")
    g.add_argument("--regiao-unica", action="store_true",
                   help="seleciona a região só uma vez (no 1º PDF) e usa em todos. "
                        "Sem esta opção, a seleção é manual em CADA PDF")
    g.add_argument("--paginas", help="ex.: 1,3-5 ou 'todas' (padrão: a página escolhida "
                                     "na seleção; com --regiao/--config, todas)")

    g = p.add_argument_group("extração")
    g.add_argument("--modo", choices=["auto", "ocr", "nativo"], default="auto",
                   help="auto: usa texto embutido se existir, senão OCR (padrão)")
    g.add_argument("--lang", default="por+eng", help="idiomas do Tesseract (padrão: por+eng)")
    g.add_argument("--dpi", type=int, default=300, help="resolução da renderização (padrão: 300)")
    g.add_argument("--psm", type=int, default=6, help="modo de segmentação do Tesseract "
                                                      "(6 = bloco; tente 4 ou 11 se falhar)")
    g.add_argument("--conf-min", type=float, default=30, help="confiança mínima do OCR (0-100)")
    g.add_argument("--manter-linhas", action="store_true",
                   help="não remove as linhas de grade da tabela antes do OCR")
    g.add_argument("--gap", type=float, default=0.7,
                   help="largura mínima do vão entre colunas, em alturas de letra (padrão: 0.7)")
    g.add_argument("--tolerancia", type=float, default=0.0,
                   help="fração de linhas que podem invadir o vão da coluna (ex.: 0.1)")
    g.add_argument("--texto", action="store_true", help="não separa em colunas: uma linha = uma célula")

    g = p.add_argument_group("saída")
    g.add_argument("--cabecalho", action="store_true", help="usa a 1ª linha como cabeçalho")
    g.add_argument("--numeros", action="store_true", help="converte números no padrão BR (1.234,56)")
    g.add_argument("--salvar-recorte", metavar="PASTA",
                   help="salva as imagens recortadas (útil para conferir a região)")
    g.add_argument("--tesseract-cmd", help="caminho do executável do Tesseract (Windows)")
    return p


def main():
    args = criar_parser().parse_args()
    modo_interativo = not args.pdfs
    if modo_interativo:
        args.pdfs, recursivo_janela = escolher_pdfs()
        args.recursivo = args.recursivo or recursivo_janela
        if not args.pdfs:
            sys.exit("Nenhum PDF selecionado.")
    args.pdfs = expandir_entradas(args.pdfs, args.recursivo)
    if not args.pdfs:
        sys.exit("Nenhum PDF encontrado nas entradas informadas.")
    for pdf in args.pdfs:
        if not Path(pdf).is_file():
            sys.exit(f"Arquivo não encontrado: {pdf}")
    print(f"{len(args.pdfs)} PDF(s) para processar.")

    # Region 1an1 PDF file
    # selecoes[caminho] = (pagina ou None, (x0, y0, x1, y1))
    selecoes: dict[str, tuple] = {}
    regiao_unica = None  # preenchido quando a mesma região vale para todos

    if args.regiao:
        regiao_unica = tuple(float(v) for v in args.regiao.split(","))
        if len(regiao_unica) != 4 or not all(0 <= v <= 1 for v in regiao_unica) \
                or regiao_unica[0] >= regiao_unica[2] or regiao_unica[1] >= regiao_unica[3]:
            sys.exit("--regiao deve ser x0,y0,x1,y1 com valores entre 0 e 1 (x0<x1, y0<y1).")
        selecoes = {c: (None, regiao_unica) for c in args.pdfs}
    elif args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if "regiao" in cfg:
            regiao_unica = tuple(cfg["regiao"])
            selecoes = {c: (None, regiao_unica) for c in args.pdfs}
        else:  # config gerada por seleção manual por arquivo
            for c in args.pdfs:
                item = cfg.get("por_arquivo", {}).get(Path(c).name)
                if item:
                    selecoes[c] = (item["pagina"], tuple(item["regiao"]))
                else:
                    print(f"AVISO: {Path(c).name} não consta em {args.config}; será ignorado.")
    elif args.regiao_unica:
        sel = selecionar_regiao_gui(args.pdfs[0], rotulo=f"[1 região p/ todos] {Path(args.pdfs[0]).name}")
        if sel is None:
            sys.exit("Seleção cancelada.")
        regiao_unica = sel[1]
        selecoes = {c: (sel[0], regiao_unica) for c in args.pdfs}
    else:
        total = len(args.pdfs)
        for n, c in enumerate(args.pdfs, 1):
            sel = selecionar_regiao_gui(c, rotulo=f"PDF {n}/{total}: {Path(c).name}")
            if sel is None:
                print(f"    {Path(c).name}: ignorado (Esc)")
            else:
                selecoes[c] = sel
        if not selecoes:
            sys.exit("Nenhuma região foi selecionada.")

    if args.salvar_config:
        if regiao_unica:
            dados_cfg = {"regiao": list(regiao_unica)}
        else:
            dados_cfg = {"por_arquivo": {Path(c).name: {"pagina": pg, "regiao": list(r)}
                                         for c, (pg, r) in selecoes.items()}}
        Path(args.salvar_config).write_text(
            json.dumps(dados_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Região salva em {args.salvar_config}")

    idioma = configurar_tesseract(args.tesseract_cmd, args.lang)
    pasta_recortes = Path(args.salvar_recorte) if args.salvar_recorte else None
    if pasta_recortes:
        pasta_recortes.mkdir(parents=True, exist_ok=True)

    # Extract
    quadros, linhas_texto, resumo = [], [], []
    for n, caminho in enumerate(args.pdfs, 1):
        nome = Path(caminho).name
        print(f"[{n}/{len(args.pdfs)}] {nome}")
        n_linhas, n_paginas = 0, 0
        if caminho not in selecoes:
            resumo.append({"arquivo": nome, "paginas_lidas": 0, "linhas_extraidas": 0,
                           "status": "Ignorado (sem região selecionada)"})
            continue
        pagina_sel, regiao = selecoes[caminho]
        try:
            doc = fitz.open(caminho)
            padrao = [pagina_sel] if pagina_sel is not None else list(range(len(doc)))
            for i in interpretar_paginas(args.paginas, len(doc), padrao):
                tabela, recorte = extrair_pagina(doc[i], regiao, args, idioma)
                n_paginas += 1
                print(f"    página {i + 1}: {len(tabela)} linha(s)")
                if pasta_recortes and recorte is not None:
                    recorte.save(pasta_recortes / f"{Path(caminho).stem}_p{i + 1}.png")
                if not tabela:
                    continue
                n_linhas += len(tabela)
                if args.texto:
                    linhas_texto += [f"[{nome} - pág. {i + 1}]"] + [l[0] for l in tabela] + [""]
                    continue
                df = pd.DataFrame(tabela)
                df.insert(0, "pagina", i + 1)
                df.insert(0, "arquivo", nome)
                quadros.append(df)
            status = "OK" if n_linhas else "Sem dados na região"
        except Exception as erro:  # um PDF com problema não interrompe os demais
            status = f"ERRO: {erro}"
            print(f"    {status}", file=sys.stderr)
        resumo.append({"arquivo": nome, "paginas_lidas": n_paginas,
                       "linhas_extraidas": n_linhas, "status": status})

    # Save
    if args.saida:
        saida = Path(args.saida)
    else:
        nome_base = ("consolidado" if len(args.pdfs) > 1 else Path(args.pdfs[0]).stem + "_extraido")
        saida = Path(args.pdfs[0]).with_name(nome_base + (".txt" if args.texto else ".xlsx"))
        if modo_interativo:
            saida = escolher_saida(saida)

    if args.texto:
        if not linhas_texto:
            sys.exit("Nada foi extraído. Confira a região com --salvar-recorte.")
        saida.write_text("\n".join(linhas_texto), encoding="utf-8")
        print(f"Salvo em {saida}")
        return

    if not quadros:
        sys.exit("Nada foi extraído. Confira a região com --salvar-recorte "
                 "ou ajuste --psm / --dpi / --conf-min.")

    df = pd.concat(quadros, ignore_index=True)
    if args.cabecalho:
        dados = df.iloc[:, 2:]
        cab = [str(c).strip() or f"col{i + 1}" for i, c in enumerate(dados.iloc[0])]
        # remove o cabeçalho repetido em cada página
        repetido = dados.apply(lambda r: [str(x).strip() for x in r] == cab, axis=1)
        df = pd.concat([df.iloc[:, :2], dados], axis=1)[~repetido]
        df.columns = list(df.columns[:2]) + cab
        df = df.reset_index(drop=True)
    if args.numeros:
        for c in df.columns[2:]:
            df[c] = df[c].map(lambda v: para_numero(v) if isinstance(v, str) else v)

    if saida.suffix.lower() == ".csv":
        df.to_csv(saida, index=False, sep=";", encoding="utf-8-sig")  # abre certo no Excel BR
    else:
        with pd.ExcelWriter(saida, engine="openpyxl") as escritor:
            df.to_excel(escritor, sheet_name="Dados", index=False)
            pd.DataFrame(resumo).to_excel(escritor, sheet_name="Resumo", index=False)
            for aba in escritor.sheets.values():  # largura de coluna legível
                for col in aba.columns:
                    maior = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                    aba.column_dimensions[col[0].column_letter].width = min(maior + 2, 50)
    ok = sum(r["status"] == "OK" for r in resumo)
    print(f"\n{len(df)} linha(s) de {ok}/{len(resumo)} PDF(s) salvas em {saida}")
    for r in resumo:
        if r["status"] != "OK":
            print(f"  ATENÇÃO — {r['arquivo']}: {r['status']}")


if __name__ == "__main__":
    interativo = len(sys.argv) == 1  # executado sem argumentos (ex.: duplo clique)
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code, file=sys.stderr)
        if interativo:
            input("\nPressione Enter para sair...")
        sys.exit(e.code if isinstance(e.code, int) else (1 if e.code else 0))
    else:
        if interativo:
            input("\nPressione Enter para sair...")