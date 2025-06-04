#!/bin/bash


if test -z $1; then
    file=$1
    pandoc -f markdown --markdown-headings=setext --output="${file::-3}.pdf" --include-in-header=header-style.tex $file
else
    md_files=$(find . | grep md$)
    for file in $md_files; do
        pandoc --pdf-engine=pdfroff --output="${file::-3}.pdf" $file
    done
    #pdfunite assignment3 *.pdf
fi
