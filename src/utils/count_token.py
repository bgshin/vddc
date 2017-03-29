
# target = 'amazon' # 542
target = 'yelp' # 1195
fname = '../../data/%s/train.tsv' % target

max_token = 0
with open(fname, 'rt') as lines:
    for line in lines:
        txt = line.split('\t')[1]
        tokens = txt.split(' ')
        ntoken = len(tokens)

        if max_token<ntoken:
            max_token = ntoken

print max_token
