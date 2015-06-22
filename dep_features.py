# -*- coding: utf-8 -*-
import codecs
import os
import csv
import sys
import logging
from collections import defaultdict

usage = 'usage: dep_features.py infile features-dir outfile'

csv.field_size_limit(sys.maxint)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

attribs_to_ignore = ['sentence', 'child', 'head',  'sentence2', 'dep_pair', 'headCheck',
                     'headToken', 'childToken', 'sent', 'dep_pairs_by_POS', 'pairs', 'true_pairs']

attribs_to_ignore = dict([(a, True) for a in attribs_to_ignore])


class Token(object):
    def __init__(self, token, SID, sentences_dict):
        """
        :param token: dict(WID=row['wordNo'], token=row['token'], head=row['head'],
                           check=check, gram=row['gram'], lemma=row['lemma'], GS_head=None)
        :param SID: sentence ID to look up in sentences_dict for

        00 in head and check means:
         initial value was '' or 'h'
        """
        self.SID = int(SID)
        self.sent = sentences_dict[SID]
        self.WID = int(token['WID'])
        self.token = token['token']
        self.GS_head = self.get_value(token['GS_head'], SID, token)
        self.head = self.get_value(token['head'], SID, token)
        self.check = self.check_tok(token['check'])
        self.gram = token['WID']
        self.lemma = token['lemma']

    @staticmethod
    def get_value(value, SID, token):
        try:
            if value == '' or value == 'h': return 00
            elif value is None or value == 'NULL': return None
            else: return int(value)
        except Exception, error:
            logging.warning('Got error: {} on value {}, sent {}'.format(error, value, SID))
            logging.warning('Token looks like: {}'.format(token))
            return value

    @staticmethod
    def check_tok(value):
            a = lambda x: x in ['0', '2', '3', '4', '5']
            if a(value): return True
            else: return False


class SentenceRule(object):
    """
    Собрать все пары <head-child> в предложении (строки предложения datatype Token)
    """
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence2 = sentence
        self.sentence_length_in_words = len(self.sentence)
        self.sentence_length_in_chars = self.sentence_len_chars()

    #TODO: preparsing - extract subtrees or NPs, VPs etc, + having WIDs as their attributes

    def sentence_len_chars(self):
        return self.len_chars(condition=None)

    def len_chars(self, condition=None):
        if condition is None:
            return sum([len(word.token) for word in self.sentence])
        else:
            return sum([len(word.token) for word in self.sentence if condition(word)])

    def get_pairs(self, childHeadfeauture=None):
        self.pairs = []
        self.true_pairs = []
        for head in self.sentence:
            for child in self.sentence2:
                if childHeadfeauture == 'GS_head':
                    if head.WID == child.GS_head:
                        self.pairs.append([head, child])
                elif childHeadfeauture == 'head':
                    if head.WID == child.head and (child.head != '' or child.head != 00):
                        self.pairs.append([head, child])
                elif childHeadfeauture == 'all':
                    if head.WID == child.head and (child.head != '' or child.head != 00):
                        self.pairs.append([head, child])
                    if head.WID == child.GS_head:
                        true_child = child
                        true_child.check = True
                        true_child.head = true_child.GS_head
                        pair = [head, true_child]
                        if pair not in self.pairs:
                            self.true_pairs.append(pair)
        if len(self.true_pairs) > 0:
            self.pairs.extend([pair for pair in self.true_pairs])
        return self.pairs

    def convert_gs_attribs(self):
        self.true_pairs = [[self.head, self.child.__setattr__('check', True)]
                                    for (self.head, self.child) in self.true_pairs]

    def get_dep_pairs_by_POS(self, POSfeature, POSasWhat):
        """Вернуть все пары зависимостей для указанного грам.тега и роли токена (head / child),
        в одном предложении
        """
        self.POSfeature = POSfeature
        self.POSasWhat = POSasWhat
        self.dep_pairs_by_POS = []
        for head in self.sentence:
            for child in self.sentence2:
                if self.POSasWhat == 'head':
                    if self.POSfeature in head.gram:
                        if child.head == head.WID:
                                self.dep_pairs_by_POS.append([head, child])
                elif self.POSasWhat == 'child':
                        if self.POSfeature in child.gram:
                            if child.head == head.WID:
                                self.dep_pairs_by_POS.append([head, child])
                elif self.POSasWhat == 'GS_head':
                    if self.POSfeature in head.gram:
                        if head.WID == child.GS_head:
                           self.dep_pairs_by_POS.append([head, child])
                elif self.POSasWhat == 'GS_child':
                    if self.POSfeature in child.gram:
                        if head.WID == child.GS_head:
                            self.dep_pairs_by_POS.append([head, child])
        return self.dep_pairs_by_POS

    @staticmethod
    def position(head, child):
        """Вернуть позицию ребёнка по отношению к главному слову
        """
        if head.WID < child.WID:  return 'pre-pos'
        elif head.WID > child.WID: return 'post-pos'


class CommonRule(SentenceRule):
    """Класс для описания связи <head - child> из данных типа <SentenceRule>."""
    def __init__(self, dep_pair, sentence, tags=None):
        SentenceRule.__init__(self, sentence)
        self.dep_pair = dep_pair
        self.head, self.child = self.dep_pair
        self.id = '%s.%s.%s' % (self.head.SID, self.head.WID, self.child.WID)
        self.headToken = self.head.token
        self.childToken = self.child.token
        self.sentence = sentence
        self.child_post_pos = self.check_position(what='post-pos')
        self.child_pre_pos = self.check_position(what='pre-pos')
        self.distance_in_words = self.distance()
        self.distance_in_chars = self.distance(chars=True)
        self.has_words_between = self.words_between(bit=True)
        self.has_words_before = self.words_before(bit=True)
        self.has_words_after = self.words_after(bit=True)
        self.words_between_are_head_children = self.words_dependencies(what=self.head, how='between', bit=True)
        self.words_between_are_child_children = self.words_dependencies(what=self.child, how='between', bit=True)
        self.words_after_are_head_children = self.words_dependencies(what=self.head, how='after', bit=True)
        self.words_after_are_child_children = self.words_dependencies(what=self.child, how='after', bit=True)
        self.words_before_are_head_children = self.words_dependencies(what=self.head, how='before', bit=True)
        self.words_before_are_child_children = self.words_dependencies(what=self.child, how='before', bit=True)
        self.child_head_is_void = self.check_is_void(self.child.head)
        self.head_head_is_void = self.check_is_void(self.head.head)
        self.sentence_length_in_words = self.sentence_length_in_words
        self.sentence_length_in_chars =  self.sentence_length_in_chars
        self.child_has_children = self.check_for_children(self.child)
        self.head_has_other_children = self.check_for_children(self.head)
        self.head_is_smth_else_child = self.check_for_head(self.head)
        self.head_is_root = self.check_root(self.head)
        self.child_is_root = self.check_root(self.child)
        self.has_punct_between = self.pos_between(bit=True, pos='pnt')
        self.has_fin_between = self.pos_between(bit=True, pos='fin')
        self.childCheck = self.child.check
        self.headCheck = self.head.check
        if tags is not None:
            self.add_gram_attribs(tags)

    def add_gram_attribs(self, tags, tag_separator=','):
        head_tags = dict([(tag, True) for tag in self.head.gram.split(tag_separator)])
        child_tags = dict([(tag, True) for tag in self.child.gram.split(tag_separator)])
        for tag in tags:
                if tag in head_tags: self.__setattr__('HEAD_is_%s' % tag, True)
                elif tag not in head_tags: self.__setattr__('HEAD_is_%s' % tag, False)
                if tag in child_tags: self.__setattr__('CHILD_is_%s' % tag, True)
                elif tag not in child_tags: self.__setattr__('CHILD_is_%s' % tag, False)

    @staticmethod
    def check_root(what):
        if what.head == 0 or what.head == '0':
            return True
        else: return False

    @staticmethod
    def check_is_void(what):
        if what == 00: return True
        else: return False

    @staticmethod
    def position(head, child):
        if head.WID < child.WID:  return 'pre-pos'
        elif head.WID > child.WID: return 'post-pos'

    def words_before(self, what=None, bit=None):
        if what is not None:
            condition = lambda word, what: word.head == what.WID \
                                           and (word.WID < self.head.WID and word.WID < self.child.WID
                                                             and word.head != '' and word.head != 00)
        else:
            condition = lambda word, what: (word.WID < self.head.WID and word.WID < self.child.WID
                                               and word.head != '' and word.head != 00)
        return self.words_at_positions(what=what, bit=bit, condition=condition)

    def words_after(self, what=None, bit=None):
        if what is not None:
            condition = lambda word, what: word.head == what.WID \
                                           and (word.WID > self.head.WID and word.WID > self.child.WID
                                           and word.head != '' and word.head != 00)
        else:
             condition = lambda word, what: (word.WID > self.head.WID and word.WID > self.child.WID
                                                 and word.head != '' and word.head != 00)
        return self.words_at_positions(what=what, bit=bit, condition=condition)

    def words_at_positions(self, what=None,  bit=None, condition=None):
        words = [word for word in self.sentence if condition(word, what)]
        if bit is None: return words
        elif words != [] and bit is not None: return True
        else: return False

    def words_between(self, bit=None,  what=None):
        if what is not None:
            condition = lambda word, what: ((word.WID < self.head.WID and word.WID > self.child.WID)
                                            or (word.WID > self.head.WID and word.WID < self.child.WID)
                                            and (word.head == what.WID and word.head != '' and word.head != 00))
        else:
            condition = lambda word, what: (((word.WID < self.head.WID and word.WID > self.child.WID)
                                           or (word.WID > self.head.WID and word.WID < self.child.WID))
                                                and word.head != '' and word.head != 00)
        return self.words_at_positions(what=what, bit=bit, condition=condition)

    def pos_between(self, bit=None, pos=None):
        words_between = self.words_between(bit=None)
        pos_list = []
        for word in list(words_between):
            if pos in word.gram:
                pos_list.append(word)
        if bit is None:
            return pos_list
        else:
            return True if len(pos_list) > 0 else False

    def words_dependencies(self, what=None, bit=None, how=None):
        condition = lambda  word: word.head == what.WID
        if how == 'between':
            words = self.words_between(what=what)
        elif how == 'after':
            words = self.words_after(what=what)
        elif how == 'before':
            words = self.words_before(what=what)
        else: words = []
        words = [word for word in words if condition(word)]
        if words != [] and bit is True:
            return True
        elif words != [] and bit is None:
            return words
        else:
            return False

    def check_position(self, what=None):
        if what == self.position(self.head, self.child): return True
        else: return False

    def distance(self, chars=False):
        if chars is False:
            return abs((int(self.head.WID) - int(self.child.WID)))
        else:
            condition = lambda word: \
                (word.WID < self.head.WID  and word.WID > self.child.WID) \
                or (word.WID > self.head.WID and word.WID < self.child.WID)
            return self.len_chars(condition=condition)

    def check_for_children(self, what):
        condition = lambda word, what:  word.head == what.WID
        if what.WID in [word.WID for word in self.sentence if condition(word, what)]:
            return True
        else: return False

    def check_for_head(self, what):
        condition = lambda what, word:  word.WID == what.head
        if what.WID in [word.WID for word in self.sentence if condition(what, word)]:
            return True
        else: return False

    @staticmethod
    def get_pos(token, *args):
        for arg in args:
            if arg in token.gram: return True
        else: return False


class ChildRule(CommonRule):
    def __init__(self, dep_pair,  tags=None):
        CommonRule.__init__(self, dep_pair, tags=None)
        self.dep_pair = dep_pair
        self.head, self.child = self.dep_pair
        self.id = '%s.%s.%s' % (self.head.SID, self.head.WID, self.child.WID)
        self.headToken = self.head.token
        self.childToken = self.child.token
        self.sentence = self.head.sent
        self.child_head_is_void = self.check_is_void(self.child.head)
        self.sentence_length_in_words = len(self.sentence)
        self.sentence_length_in_chars = self.sentence_len_chars()
        self.child_has_children = self.check_for_children(self.child)
        self.child_is_root = self.check_root(self.child)
        self.childCheck = self.child.check
        if tags is not None:
            self.add_gram_attribs(tags, what='CHILD')

    def add_gram_attribs(self, tags, tag_separator=',', what=None):
        if what == 'HEAD':
            what_tags = dict([(tag, True) for tag in self.head.gram.split(tag_separator)])
        elif what == 'CHILD':
            what_tags = dict([(tag, True) for tag in self.child.gram.split(tag_separator)])
        for tag in tags:
                if tag in what_tags: self.__setattr__('%s_is_%s' % (what, tag), True)
                elif tag not in what_tags: self.__setattr__('%s_is_%s' % (what, tag), False)


def read_file(filename, tags=None, delimiter=';'):
    with open(filename, 'rb') as infile:
        csv_reader = csv.DictReader(infile, delimiter=delimiter)
        cur_sent_id = None
        results = defaultdict(list)
        tags_set = list()
        links = list()
        sent_dict = defaultdict(list)
        for row in csv_reader:
            if not row['SID']:
                    row['SID'] = cur_sent_id
            else:
                    cur_sent_id = row['SID']
            if not row['SA/Check']: check = 1
            else: check = row['SA/Check']
            sent_dict[cur_sent_id].append(row['token'])
            results[cur_sent_id].append(dict(WID=row['WID'], token=row['Token'], head=row['SA/Head'], check=check,
                                             gram=row['SA/Gramm'], lemma=row['Lemma'], GS_head=row['GS/Head']))

            if tags is True:
                for tag in row['SA/Gramm'].split(','):
                    if tag not in tags_set and tag != '':
                        tags_set.append(tag)
                if row['Type'] not in links:
                    links.append(row['Type'])

    return results, tags_set, links, sent_dict


def read_it(filename):
    results = list()
    links = list()
    tags_set = list()
    with open(filename, 'rb') as infile:
        dialect = csv.Sniffer().sniff(infile.read(1024))
        delimiter = dialect.delimiter
        infile.seek(0)
        csv_reader = csv.DictReader(infile, delimiter=delimiter)
        headers = csv_reader.fieldnames
    if 'SID' in headers:
        results, tags_set, links, sent_dict = read_file(filename, tags=True, delimiter=delimiter)
    elif 'sentNo' in headers:
        results, tags_set, links, sent_dict = read_markup(filename, delimiter=delimiter)
    if not results:
        logging.error("Couldn't read the file specified ({}). Stopping...".format(filename))
        sys.exit()
    return results, tags_set, links, sent_dict

def read_markup(filename, tags=True, delimiter=';'):
    with open(filename, 'rb') as infile:
        csv_reader = csv.DictReader(infile, delimiter=delimiter)
        headers = csv_reader.fieldnames
        row = dict([(a, headers.index(a)) for a in headers])
        cur_sent_id = None
        results = defaultdict(list)
        tags_set = list()
        links = list()
        sent_dict = defaultdict(list)
        # for row in csv_reader:
        for line in infile:
            line = line.strip()
            line = line.split(delimiter)
            if not line[row['sentNo']]:
                    line[row['sentNo']] = cur_sent_id
            else:
                    cur_sent_id = line[row['sentNo']]
            check = 1
            sent_dict[cur_sent_id].append(line[row['token']])
            results[cur_sent_id].append(dict(WID=line[row['wordNo']], token=line[row['token']],
                                                 head=line[row['head']], check=check, gram=line[row['gram']],
                                                 lemma=line[row['lemma']], GS_head=None))
            if tags:
                for tag in line[row['gram']].split(','):
                    if tag not in tags_set and tag != '':
                        tags_set.append(tag)
                if line[row['type']] not in links:
                    links.append(line[row['type']])

    return results, tags_set, links, sent_dict


def build_features(in_fn, param='all'):
    logging.info('Reading file {} '.format(in_fn))
    results, tags, links, sent_dict = read_it(in_fn)
    logging.info('Extracting features')
    token_sentences = [[Token(feat, sent, sent_dict) for feat in results[sent]] for n, sent in enumerate(results)]
    genius_move = [CommonRule(dep_pair=dep_pair, tags=tags, sentence=sentence)
                   for sentence in token_sentences
                   for dep_pair in SentenceRule(sentence).get_pairs(param)]
    return genius_move


def save_extracted_feats(outfn, outdir, results):
    logging.info('Saving extracted features in {}'.format(outfn))
    condition = lambda x, y: (x not in y)
    separator = ','
    with codecs.open(os.path.join(outdir, outfn), 'w') as outfile:
        for n, coup_de_genie in enumerate(results):
                keys = coup_de_genie.__dict__.keys()
                values = coup_de_genie.__dict__.values()
                if n == 0: outfile.write((separator.join([a for a in keys
                                                          if condition(a, attribs_to_ignore)])) + '\n')
                outfile.write(separator.join([str(b) for (a, b)
                                              in zip(keys, values)
                                              if condition(a, attribs_to_ignore)]) + '\n')

def get_features(infile, featsdir, docname, param='all'):
    results = build_features(infile, param)
    save_extracted_feats(outdir=featsdir, outfn=docname, results=results)

def _profile_it(in_fn, featsdir, param, docname):
    import cProfile
    import pstats
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    cProfile.run("build_features('%s', param='%s')" % (in_fn, param), 'cProfile.tmp')
    c = pstats.Stats('cProfile.tmp')
    c.sort_stats('tottime').print_stats(50)
    with PyCallGraph(output=GraphvizOutput()):
        results = build_features(in_fn=in_fn, param=param)
        save_extracted_feats(outdir=featsdir, outfn=docname, results=results)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stdout.write('\n{}'.format(usage))
        sys.exit()

    in_fn = sys.argv[1]
    featsdir = sys.argv[2]
    docname = sys.argv[3]
    get_features(infile=in_fn, featsdir=featsdir, docname=docname, param='head')







