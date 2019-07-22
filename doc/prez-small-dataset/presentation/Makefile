
all: presentation.pdf
%.pdf: %.md
	rm -f presentation.pdf
	pandoc -t beamer --pdf-engine=xelatex -V -fmarkdown-implicit_figures  presentation.md -o presentation.pdf -V theme:Boadilla -V colortheme:Dolphin
clean:
	rm -f presentation.pdf
