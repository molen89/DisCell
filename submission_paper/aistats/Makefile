.PHONY: all clean
all: main.pdf
main.pdf: main.tex macros.tex references.bib sections/*.tex appendix/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -C main.tex
