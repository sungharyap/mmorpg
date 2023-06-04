from utils import get_ranking

# start date
DATE1 = 20210101
# end date
DATE2 = 20210228
# temporarily save

ranking_list = get_ranking(DATE1, DATE2)

with open(f"{DATE1}_{DATE2}.txt", 'w') as f:
    for i in ranking_list:
        f.write(i)
        f.write('\n')
