#!/bin/bash
list="gcn gat sage"
datasets="cora citeseer pubmed airport"
for backbone in $list
do
    for data in $datasets
    do
        source ./scripts/NC/"$backbone"/"$data".sh
    done
done