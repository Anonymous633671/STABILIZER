# python src/attribute_selector.py
python src/birch_bellwether_p_CFS.py
python src/Find_bellwethers.py
python src/Performance_Calculator.py
python src/Stats_files.py

echo "=========monthly_closed_PRs========="
cat results/Stats/monthly_closed_PRs.txt | python2 Stats_med.py --text 30
echo "=========monthly_closed_issues========="
cat results/Stats/monthly_closed_issues.txt | python2 Stats_med.py --text 30
echo "=========monthly_commits========="
cat results/Stats/monthly_commits.txt | python2 Stats_med.py --text 30
echo "=========monthly_contributors========="
cat results/Stats/monthly_contributors.txt | python2 Stats_med.py --text 30
echo "=========monthly_open_PRs========="
cat results/Stats/monthly_open_PRs.txt | python2 Stats_med.py --text 30
echo "=========monthly_open_issues========="
cat results/Stats/monthly_open_issues.txt | python2 Stats_med.py --text 30
echo "=========monthly_stargazer========="
cat results/Stats/monthly_stargazer.txt | python2 Stats_med.py --text 30