# Proposal Compilation Instructions

This directory contains the LaTeX source code for the proposal.

## Prerequisites

Ensure you have a TeX distribution installed (e.g., TeX Live, MacTeX, or MiKTeX).

## Compilation

### Option 1: Using `latexmk` (Recommended)

`latexmk` automatically handles dependencies (like bibliography and cross-references) and reruns LaTeX as many times as necessary.

To compile the PDF:

```bash
latexmk -pdf proposal.tex
```

To clean up auxiliary files after compilation:

```bash
latexmk -c
```

### Option 2: Manual Compilation

If you do not have `latexmk`, you can run the standard sequence of commands:

1.  Compile the document:
    ```bash
    pdflatex proposal.tex
    ```
2.  Generate the bibliography:
    ```bash
    bibtex proposal
    ```
3.  Compile again to link references:
    ```bash
    pdflatex proposal.tex
    ```
4.  Compile a final time to resolve all cross-references:
    ```bash
    pdflatex proposal.tex
    ```

## Output

The compiled file will be named `proposal.pdf`.
