from pyspark import SparkContext, SparkConf

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

    sc.textFile('paralText/word_alignments.txt') \
        .map(lambda line: extract(line)) \
        .filter(lambda x: Jaccard(x[0], x[1]) > 0.5) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) < 5) \
        .flatMapValues(lambda x: x) \
        .saveAsTextFile('paralText/input')
if __name__ == "__main__":
        main()
