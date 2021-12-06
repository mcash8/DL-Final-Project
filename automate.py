import os


'''
with open('aggregated_result.txt', 'w', encoding='utf-8') as f:
    lines_set = set()
    for folder in os.listdir('epochs'):
        with open('epochs/%s/result_predict.txt' % folder, encoding='utf-8') as fin:
            for line in fin.readlines():
                lines_set.add(line)
    f.writelines(lines_set)
'''

dicto = {}
scores = []
with open('bullshit.txt', encoding='utf-8') as bs:
    for line in bs.readlines():
        lin, score = line.rstrip().split(':')
        dicto[lin] = float(score)

for folder in os.listdir("epochs"):
    score = 0
    with open('epochs/%s/result_predict.txt' % folder, encoding='utf-8') as fin:
        lines = fin.readlines()
        assert len(lines) == 233
        for line in lines:
            score += dicto[line.rstrip()]
        scores.append((score / 233, folder))
best = (100, "scores suck. shit be broke")
for pair in scores:
    if pair[0] < best[0]:
        best = pair
print(best)