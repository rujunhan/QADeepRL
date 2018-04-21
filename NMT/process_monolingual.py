from pyspark import SparkContext, SparkConf
import random
### import spark context                                                                                                                                                                                                         
App_Name = "PARALTEXT"
conf = SparkConf().setAppName(App_Name)
sc = SparkContext(conf=conf)


def Jaccard(str1, str2):

    str1 = str1.split(' ')
    str2 = str2.split(' ')
    intersection = set(str1).intersection(set(str2))
    union = set(str1).union(set(str2))
    return (float(len(intersection)) / float(len(union)))

def extract(line):
    
    line = line.strip().split('\t')

    # get rid of the question mark
    return [line[0][:-1], line[1][:-1]]

def main():

    data = sc.textFile('paralText/word_alignments.txt') \
        .map(lambda line: extract(line)) \
        .filter(lambda x: Jaccard(x[0], x[1]) > 0.5) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) < 5) \
        .flatMapValues(lambda x: x) \
        .collect()
    
    # split data in to train / val ; source / target
    # save data from hdfs to local
    source_trn = open("input/source_trn_m.txt", 'w')
    target_trn = open("input/target_trn_m.txt", 'w')
    source_val = open("input/source_val_m.txt", 'w')
    target_val = open("input/target_val_m.txt", 'w')
    
    random.shuffle(data)
    for i in data:
        thresh = random.randint(0, 3)
        if thresh == 3:
            source_val.write('<2en> ' + i[0].encode('ascii','ignore')[:-1] + '\n')
            target_val.write(i[1].encode('ascii','ignore')[:-1] + '\n')
        else:
            source_trn.write('<2en> ' + i[0].encode('ascii','ignore')[:-1] + '\n')
            target_trn.write(i[1].encode('ascii','ignore')[:-1] + '\n')

    source_trn.close()
    target_trn.close()
    source_val.close()
    target_val.close()
if __name__ == "__main__":
        main()
