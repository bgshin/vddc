from tqdm import tqdm
import argparse

def run_to_tsv(target):
    # # target = 'test'
    # target = 'train'
    # # target = 'sample'

    special_token = 'JHFVJHGLUGIUTUDYUCUHYVJUV'

    fname = '../../../data/amazon/%s.csv' % target
    # fname = '../data/sample.csv'
    # ./bin/nlpdecode -c config-decode-en.xml -i input.txt -oe nlp2

    cls_list = []
    print 'reading contents...'
    with open(fname, 'rt') as lines:
        for line in lines:
            tokens = line.split('","')
            cls = tokens[0].replace('"','')
            # title = tokens[1].replace('""','"')
            # contents = tokens[2].replace('""','"').replace('\n','')[:-1]
            # title_contents = '%s. %s' % (title, contents)
            cls_list.append(cls)


    if target=='train':
        # N = 3600003
        num_split = 20
        fname_out = '../../../data/amazon/%s.tsv' % target
        idx = 0
        with open(fname_out, 'wt') as outhandle:
            for i in tqdm(range(num_split)):
                fname = '../../../data/amazon/%s_%02d.txt.nlp'% (target, i)
                word_list = []
                with open(fname, 'rt') as lines:
                    for line in lines:
                        if len(line)<2:
                            continue

                        tokens = line.split('\t')

                        if tokens[1]==special_token:
                            tokenized = ' '.join(word_list)
                            outline = '%s\t%s\n' % (cls_list[i], tokenized)
                            # print outline
                            outhandle.write(outline)
                            # print '\n'
                            word_list = []
                            idx+=1

                        else:
                            word_list.append(tokens[1])
                    # print tokens[1]

        print idx

    else:
        fname = '../../../data/amazon/%s.txt.nlp' % target
        fname_out = '../../../data/amazon/%s.tsv' % target
        i = 0
        word_list = []
        with open(fname_out, 'wt') as outhandle:
            with open(fname, 'rt') as lines:
                for line in lines:
                    if len(line) < 2:
                        continue

                    tokens = line.split('\t')

                    if tokens[1] == special_token:
                        tokenized = ' '.join(word_list)
                        outline = '%s\t%s\n' % (cls_list[i], tokenized)
                        # print outline
                        outhandle.write(outline)
                        # print '\n'
                        word_list = []
                        i += 1

                    else:
                        word_list.append(tokens[1])
                        # print tokens[1]

        print i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='test', type=str)
    args = parser.parse_args()

    run_to_tsv(args.t)
