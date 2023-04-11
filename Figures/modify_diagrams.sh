#!/bin/bash

# Script for modifying math in the dia-diagrams.
sed -i 's/{Lemma (bddness}/{Lemma (bounded-}/' Diagram_002.tex
sed -i 's/{of the Hks)}/{ness of \$\\norm{H_k}\$)}/' Diagram_002.tex
sed -i 's/{gk to 0}/{\$g_k\\xrightarrow{k\\to\\infty}0\$}/' Diagram_002.tex
sed -i 's/{||xk-y|| where}/{\$\\norm{x_k-y}\$ where}/' Diagram_002.tex
sed -i 's/{y is a fixed point}/{\$y\$ is a fixed point}/' Diagram_002.tex
sed -i 's/{xk to a fixed point}/{\$x_k\$ to a fixed point}/' Diagram_002.tex

sed -i 's/\\\$/\$/g' *.tex
sed -i 's/\\\_/\_/g' *.tex
sed -i 's/\\{/{/g' *.tex
sed -i 's/\\}/}/g' *.tex
sed -i 's/\\ensuremath{\\backslash}/\\/g' *.tex


