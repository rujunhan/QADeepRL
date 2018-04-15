import numpy as np
from read_data import DataSet
#from my.utils import argmax
from utils import get_phrase, get_best_span, span_f1


class Evaluation(object):

    def __init__(self, data_type, idxs):
        self.data_type = data_type
        self.idxs = idxs
        #self.yp = yp
        self.num_examples = len(idxs)
        self.dict = {'data_type': data_type,
                     #'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}

class F1Evaluation(Evaluation):

    def __init__(self, data_type, idxs, correct, loss, f1s, id2answer_dict):
        super(F1Evaluation, self).__init__(data_type, idxs)
        #self.y = y
        #self.dict['y'] = y
        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        #self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        #self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
        self.id2answer_dict = id2answer_dict

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):

        assert self.data_type == other.data_type
        new_idxs = self.idxs + other.idxs
        #new_yp = self.yp + other.yp
        #new_yp2 = self.yp2 + other.yp2
        #new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_f1s = self.f1s + other.f1s
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        return F1Evaluation(self.data_type, new_idxs, new_correct, new_loss, new_f1s, new_id2answer_dict)

    def __repr__(self):
        return "{} accuracy={:.4f}, f1={:.4f}, loss={:.4f}".format(self.data_type, self.acc, self.f1, self.loss)

class Evaluator(object):
   
    def __init__(self, conifg, model):
        self.config = config
        self.model = model
        #self.global_step = global_step
        #self.yp = yp


class F1Evaluator(object):
    
    def __init__(self, config, model):
        #super(F1Evaluator, self).__init__(config, model)
        #self.yp2 = model.yp2
        #self.loss = model.loss
        self.model = model
        self.config = config

    def get_evaluation(self, batch):
        #print(batch[0])
        idxs, data_set = self._split_batch(batch[0])
        
        assert isinstance(data_set, DataSet)

        self.model(batch)

        yp, yp2, loss = self.model.yp, self.model.yp2, self.model.build_loss()
        
        y = data_set.data['y']

        if self.config.squash:
            new_y = []

        if self.config.single:
            new_y = []

        spans = []
        scores = []

        for b in range(yp.size()[0]):
            span, score = get_best_span(yp[b].data.cpu().numpy(), yp2[b].data.cpu().numpy())
            spans.append(span)
            scores.append(score)
        #print(spans, scores)

        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        def _get2(context, xi, span):
            if len(xi) <= span[0][0]:
                return ""
            if len(xi[span[0][0]]) <= span[1][1]:
                return ""
            return get_phrase(context, xi, span)
        
        #print(list(zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p']))[0])
        #print(list(zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p']))[1])
        #print(list(zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p']))[2])

        #print(spans)

        # test case
        #spans = [((0, 7), (0, 20)), ((0, 0), (0, 2)), ((0, 700), (0,702))]
        
        #print(spans)

        id2answer_dict = {id_: _get2(context, xi, span)
                          for id_, xi, span, context in zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p'])}
        id2score_dict = {id_: score for id_, score in zip(data_set.data['ids'], scores)}
        id2answer_dict['scores'] = id2score_dict
        correct = [self.__class__.compare2(yi, span) for yi, span in zip(y, spans)]
        f1s = [self.__class__.span_f1(yi, span) for yi, span in zip(y, spans)]
        
        #print(id2answer_dict)
        #print(id2score_dict)
        #print(correct)
        #print(f1s)
        
        e = F1Evaluation(data_set.data_type, idxs, correct, float(loss), f1s, id2answer_dict)
        return e


    def get_evaluation_from_batches(self, batches):        
        #print("*"*50)
        #print(batches)
        #print("*"*50)
        #for batch in batches:
        #    print(batch)
        e = sum(self.get_evaluation(batch) for batch in batches)
        return e

    def _split_batch(self, batch):
        return batch

    @staticmethod
    def compare2(yi, span):
        for start, stop in yi:
            if tuple(start) == span[0] and tuple(stop) == span[1]:
                return True
        return False

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1

class GPUF1Evaluator(F1Evaluator):
    def __init__(self, config, model):
        super(GPUF1Evaluator, self).__init__(config, model)
        self.model = model
        N, M, JX = config.batch_size, config.max_num_sents, config.max_sent_size
        self.yp = model.yp
        self.yp2 = model.yp2
        self.loss = model.loss
