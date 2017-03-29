from tqdm import tqdm
import argparse


def run_trim(target):
    special_token = 'JHFVJHGLUGIUTUDYUCUHYVJUV'
    N_test = 400000
    N_train = 3600000
    # N_train = 40
    fname = '../../../data/amazon/%s.csv' % target

    contents_list = []
    print 'reading contents...'
    with open(fname, 'rt') as lines:
        for line in lines:
            tokens = line.split('","')
            cls = tokens[0].replace('"','')
            title = tokens[1].replace('""','"')
            contents = tokens[2].replace('""','"').replace('\n','')[:-1]
            title_contents = '%s. %s' % (title, contents)
            contents_list.append(title_contents)
            # print '%s\t%s' % (cls, title_contents)

    print 'writing to files...'

    if target=='train' or target=='sample':
        num_split = 20
        N = N_train/num_split
        for i in tqdm(range(num_split)):
            fname = '../../../data/amazon/%s_%02d.txt' % (target, i)
            with open(fname, 'wt') as handle:
                contents_list_part = contents_list[i*N:(i+1) *N]
                for idx in tqdm(range(len(contents_list_part))):
                    c = contents_list_part[idx]
                    handle.write(c)
                    handle.write('\r\n%s\r\n' % special_token)

    else:
        fname = '../../../data/amazon/%s.txt'% (target)
        with open(fname, 'wt') as handle:
            for idx in tqdm(range(len(contents_list))):
                c = contents_list[idx]
                handle.write(c)
                handle.write('\r\n%s\r\n' % special_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='train', type=str)
    args = parser.parse_args()

    run_trim(args.t)
