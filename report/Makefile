TEX_ENTRY := main.tex

.PHONY: all
all: windows

.PHONY: target
target:
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	$(MAKE) bib
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	cp output/target.pdf ./target.pdf

.PHONY: bib
bib:
	cp *.bib output
	cd output && bibtex target && cd ..

.PHONY: clean
clean:
	rm -rf target.pdf output/

.PHONY: once
once:
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	cp output/target.pdf ./target.pdf

.PHONY: rebuild
rebuild: clean all

.PHONY: windows
windows:
	-rd /s /q output
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	copy *.bib output /y
	cd output && bibtex target && cd ..
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	pdflatex -output-directory=output -jobname=target -quiet $(TEX_ENTRY)
	copy output\target.pdf target.pdf
