"""Build the EAAI submission LaTeX (double-anonymized manuscript + separate title page) from index.html.

Pipeline: index.html -> project preprocessing (contributions + Figure 1 moved into the Introduction, identifying
back matter removed, references tagged) -> html2tex skill stage 1 (convert_to_tex.py, single column, numeric) ->
numeric links rewritten to natbib author-year citations against references.bib -> elsarticle [preprint,12pt,
authoryear] + elsarticle-harv. Output: latex_eaai/manuscript.tex (+ title_page.tex), compiled with lualatex+bibtex.
EAAI rules applied (Guide for Authors, 2026): single column, <=250-word abstract stating the AI contribution and the
engineering application, no undefined acronyms in title/abstract, 1-6 keywords, author-year references,
double-anonymized review with a separate title page, generative-AI declaration before the references.
"""
import os, re, html, json, shutil, subprocess, sys
import bibtexparser
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TWOCOL = "--twocol" in sys.argv   # two-column, non-anonymized author copy (Overleaf); default: anonymized single-column EAAI submission
IJPHM = "--ijphm" in sys.argv     # International Journal of Prognostics and Health Management: PHMSociety.cls, two columns, APA, single-blind
ARXIV = "--arxiv" in sys.argv     # generic arXiv preprint: article class, two columns, with authors, natbib author-year
AUTH = TWOCOL or ARXIV or IJPHM
OUT = os.path.join(ROOT, "latex_twocol" if TWOCOL else ("latex_arxiv" if ARXIV else ("latex_ijphm" if IJPHM else "latex_eaai"))); SKILL = os.path.expanduser("~/.claude/skills/html2tex")
TITLE = "Driver or Fueling Fault? A Fuel-Trim Fault Screen with a Driving-Style Indicator for Vehicle Fleets Without Joint Cause Labels"
KEYWORDS = ["fleet health monitoring", "fault isolation", "fuel trim", "on-board diagnostics", "driver behaviour", "selective classification"]
h = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()

# ---------- abstract ----------
ab = re.search(r'<p class="lead">(.*?)</p>', h, re.S).group(1)
ab = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", ab))).strip()
nw = len(ab.split()); assert nw <= (400 if AUTH else 250), f"abstract {nw} words"
assert AUTH or not re.findall(r"\b[A-Z]{2,}\b", ab), "acronym in abstract (EAAI rule)"
ab_tex = ab.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

# ---------- body preprocessing ----------
contrib = re.search(r'<h3>Contributions</h3>\s*(<ul class="contrib">.*?</ul>)', h, re.S).group(1)
fig1 = re.search(r'<figure>\s*<img src="figures/overview\.svg".*?</figure>', h, re.S).group(0)
start = h.index('<h2 id="intro">'); refs_h = h.index('<h2 id="refs">'); end = h.index("</ol>", refs_h) + 5
APP_HTML = h[h.index('<h2 id="appendix">'):h.index('<div class="foot">')] if '<h2 id="appendix">' in h else ""
body = h[start:end] + APP_HTML
i_rel = body.index('<h2 id="related">')   # end of the Introduction
body = body[:i_rel] + "<p>The contributions of this work are:</p>\n" + contrib + "\n" + fig1 + "\n" + body[i_rel:]
if not AUTH:
    body = re.sub(r'<div class="identifying">.*?</div>', "", body, flags=re.S)
    body = re.sub(r"are in the project\s+repository", "are in a public repository (link withheld for anonymized review)", body)
body = re.sub(r'(<h2 id="refs">References</h2>\s*)<ol[^>]*>', r'\1<ol class="refs">', body)
assert '<ol class="refs">' in body
# reference entries may name the authors (third-person self-citation is allowed); the prose may not
if not AUTH:
    assert not re.search(r"Apartsin|Aperstein|Holon|Afeka|ApartsinProjects|github\.io", body.split("<ol class=\"refs\">")[0]), "identifying text in body"
src = os.path.join(ROOT, "_eaai_src.html")
open(src, "w", encoding="utf-8").write("<!doctype html><html><head><meta charset='utf-8'><title>x</title></head><body>" + body + "</body></html>")

# ---------- html2tex stage 1 ----------
if os.path.isdir(OUT): shutil.rmtree(OUT, ignore_errors=True)   # a lingering TeX run may lock a .log
r = subprocess.run([sys.executable, "-P", os.path.join(SKILL, "scripts", "convert_to_tex.py"), "--input", src, "--out-dir", OUT, "--columns", "2" if (TWOCOL or ARXIV or IJPHM) else "1"],
                   capture_output=True, text=True); print(r.stdout[-800:], r.stderr[-800:]); os.remove(src)
assert r.returncode == 0

# ---------- citations: numeric links -> natbib author-year keys ----------
db = bibtexparser.load(open(os.path.join(ROOT, "references.bib"), encoding="utf-8")); E = db.entries
norm = lambda t: re.sub(r"[^a-z0-9]", "", html.unescape(re.sub(r"<[^>]+>", "", t)).lower())
KEY = {}
for n, li in re.findall(r'<li id="r(\d+)">(.*?)</li>', h, re.S):
    doi = re.search(r'doi\.org/([^"<\s]+)', li); hit = None
    for e in E:
        if doi and e.get("doi", "").lower() == html.unescape(doi.group(1)).lower(): hit = e["ID"]; break
    if not hit:
        for e in E:
            t = norm(e.get("title", "").replace("{", "").replace("}", ""))
            if len(t) > 15 and t[:40] in norm(li): hit = e["ID"]; break
    assert hit, f"ref {n} not in references.bib"; KEY[n] = hit
bt = open(os.path.join(OUT, "body.tex"), encoding="utf-8").read()
# html2tex already split the appendix into appendix.tex; re-attach it so it gets the same citation and heading
# handling, then it is split off again below (bug fixed 2026-10-18: the appendix was being overwritten with "").
_conv_app = os.path.join(OUT, "appendix.tex")
if os.path.exists(_conv_app):
    _app = open(_conv_app, encoding="utf-8").read().strip()
    if _app:
        if not re.search(r"\\section\*?\{Appendix", _app): _app = "\\section{Appendix}\n" + _app
        bt = bt.rstrip() + "\n\n" + _app + "\n"
assert ("<h2 id=\"appendix\">" not in h) or re.search(r"\\section\*?\{Appendix", bt), "appendix lost in conversion"
bt = re.sub(r"\{\[\}\s*((?:\\hyperref\[r\d+\]\{\d+\}\s*[,;]?\s*)+)\{\]\}",
            lambda m: "\\citep{" + ",".join(KEY[k] for k in re.findall(r"\\hyperref\[r(\d+)\]", m.group(1))) + "}", bt)
bt = re.sub(r"\\cite\{([^}]*)\}", lambda m: "\\citep{" + ",".join(KEY[k.strip()[3:]] for k in m.group(1).split(",")) + "}", bt)
bt = re.sub(r"\\hyperref\[r(\d+)\]\{\d+\}", lambda m: "\\citet{" + KEY[m.group(1)] + "}", bt)
assert "\\hyperref[r" not in bt
# heading numbers are typed in the HTML ("8.5 Robustness"); LaTeX numbers them itself
bt = re.sub(r"\\((?:sub)*section)\*?\{\d+(?:\.\d+)*\.?\s+", r"\\\1{", bt)
bt = re.sub(r"\\section\*?\{References\}(\\label\{[^}]*\})?\s*", "", bt)  # bibliography prints its own heading
bt = bt.replace(r"\section{Data and code availability}", r"\section*{Data and code availability}")  # unnumbered back matter
APP_TEX = ""
m_app = re.search(r"\\section\*?\{Appendix\}(\\label\{[^}]*\})?", bt)
if m_app:
    APP_TEX = bt[m_app.end():]; bt = bt[:m_app.start()]
    APP_TEX = re.sub(r"\\subsection\*?\{([A-G])\.\s+", r"\\subsection*{\1. ", APP_TEX)
    # the converter strips the "A." prefixes: re-letter unnumbered appendix subsections in order (A, B, C, ...)
    _cnt = iter("ABCDEFGHIJKL")
    APP_TEX = re.sub(r"\\subsection\{(?![A-J]\.\s)", lambda m: "\\subsection*{" + next(_cnt) + ". ", APP_TEX)
    APP_TEX = re.sub(r"\\subsubsection\*?\{", r"\\subsubsection*{", APP_TEX)
    APP_TEX = "\\section*{Appendix}\n" + APP_TEX
if IJPHM:   # apacite: parenthetical \cite, narrative \citeA
    bt, APP_TEX = [re.sub(r"\\citep\{", r"\\cite{", x) for x in (bt, APP_TEX)]
    bt, APP_TEX = [re.sub(r"\\citet\{", r"\\citeA{", x) for x in (bt, APP_TEX)]
def _brk(m):   # let a URL shown as link text break after each slash (two-column layouts)
    t = m.group(2)
    if " " in t or "/" not in t or "." not in t.split("/")[0]: return m.group(0)
    return "\\href{%s}{%s}" % (m.group(1), t.replace("/", "/\\allowbreak "))
bt, APP_TEX = [re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", _brk, x) for x in (bt, APP_TEX)]
open(os.path.join(OUT, "appendix.tex"), "w", encoding="utf-8").write(APP_TEX)
open(os.path.join(OUT, "body.tex"), "w", encoding="utf-8").write(bt)
shutil.copy(os.path.join(ROOT, "references.bib"), OUT)
open(os.path.join(OUT, ".gitignore"), "w").write("".join(x + chr(10) for x in ["*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.spl", "bib.log"]))
shutil.copy(os.path.join(SKILL, "templates", "_common", "html2tex_compat.tex"), OUT)

PRE = r"""\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{hyperref}
\input{html2tex_compat.tex}
\usepackage{newtxmath}
\providecommand{\lt}{<}
\raggedbottom
\journal{Engineering Applications of Artificial Intelligence}
\begin{document}
"""
man = PRE + r"""\begin{frontmatter}
\title{%TITLE%}
\begin{abstract}
%ABSTRACT%
\end{abstract}
\begin{keyword}
%KW%
\end{keyword}
\end{frontmatter}
\input{body.tex}
\bibliographystyle{elsarticle-harv}
\bibliography{references}
\input{appendix.tex}
\end{document}
"""
man = man.replace("%TITLE%", TITLE).replace("%ABSTRACT%", ab_tex).replace("%KW%", " \\sep ".join(KEYWORDS))
open(os.path.join(OUT, "manuscript.tex"), "w", encoding="utf-8").write(man)
if TWOCOL:
    PRE2 = r"""\documentclass[3p,twocolumn,authoryear]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{hyperref}
\usepackage{etoolbox}
\input{html2tex_compat.tex}
\usepackage{newtxmath}
\providecommand{\lt}{<}
\flushbottom
\AtBeginDocument{\flushbottom\setlength{\parskip}{0pt plus 1pt}}
\makeatletter
\renewcommand\section{\@startsection{section}{1}{\z@}{12\p@ \@plus 4\p@ \@minus 4\p@}{6\p@ \@plus 4\p@ \@minus 3\p@}{\normalsize\bfseries\boldmath}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{8\p@ \@plus 4\p@ \@minus 4\p@}{2\p@ \@plus 4\p@ \@minus 3\p@}{\normalfont\normalsize\itshape}}
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{8pt plus 0pt}\setlength{\@fpbot}{0pt plus 1fil}
\setlength{\@dblfptop}{0pt}\setlength{\@dblfpsep}{8pt plus 0pt}\setlength{\@dblfpbot}{0pt plus 1fil}
\patchcmd{\pprintMaketitle}{\clearpage}{\clearpage\vspace*{-2.2cm}}{}{}
\makeatother
\setlength{\intextsep}{8pt}\setlength{\textfloatsep}{8pt}\setlength{\floatsep}{8pt}
\journal{Engineering Applications of Artificial Intelligence}
\begin{document}
"""
    two = PRE2 + r"""\begin{frontmatter}
\title{%TITLE%}
\author[hit]{Alexander Apartsin\corref{cor1}}
\ead{apartsin@gmail.com}
\author[afeka]{Yehudit Aperstein}
\cortext[cor1]{Corresponding author.}
\affiliation[hit]{organization={School of Computer Science, Faculty of Sciences, Holon Institute of Technology},city={Holon},country={Israel}}
\affiliation[afeka]{organization={Intelligent Systems, Afeka Academic College of Engineering},city={Tel Aviv},country={Israel}}
\begin{abstract}
%ABSTRACT%
\end{abstract}
\begin{keyword}
%KW%
\end{keyword}
\end{frontmatter}
\input{body.tex}
\bibliographystyle{elsarticle-harv}
\bibliography{references}
\input{appendix.tex}
\end{document}
""".replace("%TITLE%", TITLE).replace("%ABSTRACT%", ab_tex).replace("%KW%", " \sep ".join(KEYWORDS))
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(two)
    os.remove(os.path.join(OUT, "manuscript.tex"))
    print("wrote latex_twocol/main.tex (two-column, with authors)")

if ARXIV:
    ARX = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage[margin=0.75in,columnsep=0.25in]{geometry}
\usepackage[authoryear,round]{natbib}
\usepackage{hyperref}
\input{html2tex_compat.tex}
\usepackage{newtxmath}
\providecommand{\lt}{<}
\raggedbottom
\title{%TITLE%}
\author{Alexander Apartsin\thanks{Corresponding author: apartsin@gmail.com}\\ \small School of Computer Science, Faculty of Sciences, Holon Institute of Technology, Holon, Israel
\and Yehudit Aperstein\\ \small Intelligent Systems, Afeka Academic College of Engineering, Tel Aviv, Israel}
\date{}
\begin{document}
\makeatletter
\twocolumn[\begin{@twocolumnfalse}
\maketitle
\begin{abstract}
%ABSTRACT%

\medskip\noindent\textbf{Keywords:} %KW%
\end{abstract}
\vspace{1em}
\end{@twocolumnfalse}]
\makeatother
\input{body.tex}
\bibliographystyle{plainnat}
\bibliography{references}
\input{appendix.tex}
\end{document}
""".replace("%TITLE%", TITLE).replace("%ABSTRACT%", ab_tex).replace("%KW%", "; ".join(KEYWORDS))
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(ARX)
    os.remove(os.path.join(OUT, "manuscript.tex"))
    print("wrote latex_arxiv/main.tex (arXiv article class, two columns, with authors)")

if IJPHM:
    shutil.copy(os.path.join(ROOT, "_downloads", "ijphm_template", "latex", "PHMSociety.cls"), OUT)
    KWL = re.search(r'<p class="kw"><b>Key ?words:</b>\s*(.*?)</p>', h, re.S | re.I); assert KWL, "keywords line not found in index.html"
    kw = html.unescape(KWL.group(1)).replace(";", ",") if KWL else ", ".join(KEYWORDS)
    IJ = r"""\documentclass[IJPHM, 2026, 00]{PHMSociety}
\usepackage{graphicx}
\usepackage{url}
\emergencystretch=2.5em
\providecommand{\href}[2]{#2}
\usepackage{amsmath}
\input{html2tex_compat.tex}
\providecommand{\lt}{<}
\begin{document}
\title{%TITLE%}
\author{%
	Alexander Apartsin\authorNumber{1} and Yehudit Aperstein\authorNumber{2}
}
\address{%
	\affiliation{1}{School of Computer Science, Faculty of Sciences, Holon Institute of Technology, Holon, Israel}{%
		{\email{apartsin@gmail.com}}
		}
	\tabularnewline
	\affiliation{2}{Intelligent Systems, Afeka Academic College of Engineering, Tel Aviv, Israel}{%
		{\email{apersteiny@afeka.ac.il}}
		}
}
\maketitle
\pagestyle{fancy}
\thispagestyle{plain}
\phmLicenseFootnote{Alexander Apartsin}
\begin{abstract}%
%ABSTRACT%

\medskip\noindent\textbf{Key Words:} %KW%
\end{abstract}
\input{body.tex}
\bibliographystyle{apacite}
\PHMbibliography{references}
\input{appendix.tex}
\end{document}
""".replace("%TITLE%", TITLE).replace("%ABSTRACT%", ab_tex).replace("%KW%", kw)
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(IJ)
    os.remove(os.path.join(OUT, "manuscript.tex"))
    print("wrote latex_ijphm/main.tex (IJPHM PHMSociety class, two columns, APA)")

# ---------- separate title page ----------
ident = re.search(r'<div class="identifying">(.*?)</div>', h, re.S).group(1)
def h2t(x):
    x = re.sub(r"<b>(.*?)</b>", lambda m: "@@B@@" + m.group(1) + "@@E@@", x, flags=re.S)
    x = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip().replace("&", r"\&")
    return x.replace("@@B@@", r"\textbf{").replace("@@E@@", "}")
paras = "\n\n".join(h2t(q) for q in re.findall(r"<p[^>]*>(.*?)</p>", ident, re.S))
tp = PRE + r"""\begin{frontmatter}
\title{%TITLE%}
\author[hit]{Alexander Apartsin\corref{cor1}}
\ead{apartsin@gmail.com}
\author[afeka]{Yehudit Aperstein}
\cortext[cor1]{Corresponding author.}
\affiliation[hit]{organization={School of Computer Science, Faculty of Sciences, Holon Institute of Technology},city={Holon},country={Israel}}
\affiliation[afeka]{organization={Intelligent Systems, Afeka Academic College of Engineering},city={Tel Aviv},country={Israel}}
\end{frontmatter}
%PARAS%
\end{document}
""".replace("%TITLE%", TITLE).replace("%PARAS%", paras)
open(os.path.join(OUT, "title_page.tex"), "w", encoding="utf-8").write(tp)
print(f"wrote latex_eaai/manuscript.tex (abstract {nw} words) and title_page.tex; {len(KEY)} citations keyed")
