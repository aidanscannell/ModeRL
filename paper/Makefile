filename=paper
# outdir=aux

# all: ${filename}.org zotero-library.bib
# all: ${filename}.org
all:
	~/Homebrew/opt/emacs-mac/Emacs.app/Contents/MacOS/Emacs --batch -l emacs-config.el ${filename}.org -f org-latex-export-to-pdf --kill
	# emacs --batch -l emacs-config.el ${filename}.org -f org-latex-export-to-pdf --kill
	# latexmk -interaction=nonstopmode -outdir=${outdir} -auxdir=${outdir} -f -output-format=pdf ${outdir}/${filename}.tex

submission:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=1 -dLastPage=10 -sOutputFile=submission.pdf ${filename}.pdf

appendix:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=11 -sOutputFile=supplement.pdf ${filename}.pdf
