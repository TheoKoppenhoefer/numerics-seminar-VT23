#!/bin/bash

sed -i 's/{Bk+ well-defined}/{\$B_{k+1}\$ well-defined}/' Diagram_001.tex
sed -i 's/{Bk invertible}/{\$B_{k}\$ invertible}/' Diagram_001.tex
# sed -i 's/{Algorithm 6}/{Algorithm \\ref{alg:aa1-prs}}/' Diagram_001.tex
sed -i 's/{Lemma (bddness}/{Lemma (bounded-}/' Diagram_001.tex
sed -i 's/{of Hks)}/{ness of \$\\norm{H_{k}}\$)}/' Diagram_001.tex
sed -i 's/{type regularisatoin)}/{type regularisation)}/' Diagram_001.tex
sed -i 's/{Iteration)}/{iteration)}/' Diagram_001.tex


sed -i 's/{Lemma (bddness}/{Lemma (bounded-}/' Diagram_002.tex
sed -i 's/{of the Hks)}/{ness of \$\\norm{H_k}\$)}/' Diagram_002.tex
sed -i 's/{gk to 0}/{\$g_k\\xrightarrow{k\\to\\infty}0\$}/' Diagram_002.tex
sed -i 's/{||xk-y|| where}/{\$\\norm{x_k-y}\$ where}/' Diagram_002.tex
sed -i 's/{y is a fixed point}/{\$y\$ is a fixed point}/' Diagram_002.tex
sed -i 's/{xk to a fixed point}/{\$x_k\$ to a fixed point}/' Diagram_002.tex
