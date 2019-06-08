
all: presentation.pdf
%.pdf: %.md
	rm -f presentation.pdf
	pandoc -t beamer --pdf-engine=xelatex -fmarkdown-implicit_figures  presentation.md -o presentation.pdf

clean:
	rm -f presentation.pdf
	
