#!/bin/bash

python src/Birch_Bellwether_v2.py
python src/Find_Bellwethers.py
python src/Final_Performance.py
python src/TCA.py
python src/TPTL.py
python src/TCA_test.py
python src/Stats.py

cat src/results/median_data_1/level_2/Stats/recall.txt| python2 src/results/median_data_1/level_2/Stats/Stats.py --text 30
cat src/results/median_data_1/level_2/Stats/precision.txt| python2 src/results/median_data_1/level_2/Stats/Stats.py --text 30
cat src/results/median_data_1/level_2/Stats/pf.txt| python2 src/results/median_data_1/level_2/Stats/Stats.py --text 30
cat src/results/median_data_1/level_2/Stats/pci_20.txt| python2 src/results/median_data_1/level_2/Stats/Stats.py --text 30
cat src/results/median_data_1/level_2/Stats/ifa.txt| python2 src/results/median_data_1/level_2/Stats/Stats.py --text 30
