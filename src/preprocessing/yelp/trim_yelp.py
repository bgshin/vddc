from tqdm import tqdm
import argparse

def run_trim(target):
    special_token = 'JHFVJHGLUGIUTUDYUCUHYVJUV'
    N_test = 38000
    N_train = 560000
    fname = '../../../data/yelp/%s.csv' % target

    contents_list = []
    print 'reading contents...'
    with open(fname, 'rt') as lines:
        for line in lines:
            tokens = line.split('","')
            cls = tokens[0].replace('"', '')
            contents = tokens[1].replace('\\""', '"').replace('\\n', '').replace('\n', '')[:-1]
            title_contents = contents
            contents_list.append(title_contents)
            # print '%s\t%s' % (cls, title_contents)

    print 'writing to files...'
    print len(contents_list)
    # exit(0)

    if target == 'train':
        num_split = 10
        N = N_train / num_split
        for i in tqdm(range(num_split)):
            fname = '../../../data/yelp/%s_%d.txt' % (target, i)

            with open(fname, 'wt') as handle:
                contents_list_part = contents_list[i * N:(i + 1) * N]
                for idx in tqdm(range(len(contents_list_part))):
                    c = contents_list[idx]
                    handle.write(c)
                    handle.write('\r\n%s\r\n' % special_token)

    else:
        fname = '../../../data/yelp/%s.txt' % (target)
        with open(fname, 'wt') as handle:
            for idx in tqdm(range(len(contents_list))):
                c = contents_list[idx]
                handle.write(c)
                handle.write('\r\n%s\r\n' % special_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='test', type=str)
    args = parser.parse_args()

    run_trim(args.t)
