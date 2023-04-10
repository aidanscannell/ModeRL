$pdf_mode = 1;        # tex -> pdf

@default_files = ('main.tex');

# $pdflatex = "xelatex %O %S";
# $pdflatex="pdflatex -interaction=nonstopmode %O %S";
# $pdflatex = 'xelatex --shell-escape %O %S';
$pdflatex = 'lualatex -interaction=nonstopmode --shell-escape %O %S';
$out_dir = '.aux';
$aux_dir = '.aux';
