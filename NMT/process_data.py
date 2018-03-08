import json
from args import parse_args


LANG_PAIR_LIST = [['en', 'zh']]

def main():

    args = parse_args()
    save_data = []

    idx = 0
    for pair in LANG_PAIR_LIST:

        with open(args.source+pair[0]+'.txt', 'r') as source:
            for line in source:
                temp = {}
                line = line.strip()
                temp['s'] = pair[0]
                temp['st'] = line
                save_data.append(temp)
        source.close()

        with open(args.source+pair[1]+'.txt', 'r') as target:
            for line in target:
                line = line.strip()
                save_data[idx]['t'] = pair[1]
                save_data[idx]['tt'] = line
                idx += 1
        target.close()

    save_file = args.source + 'en_zh.json'
    with open(save_file, 'w') as fout:
        json.dump(save_data, fout)
            

if __name__ == '__main__':
    main()

