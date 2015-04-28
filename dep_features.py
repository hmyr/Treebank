# -*- coding: utf-8 -*-
import codecs
import os
import csv
import sys

attribs_to_ignore = ['sentence', 'child', 'head',  'sentence2', 'dep_pair', 'headCheck',
                     'headToken', 'childToken', 'sent']

attribs_to_ignore = dict([(a, True) for a in attribs_to_ignore])


class Token(object):
    def __init__(self, token, SID):
        """
        00 in head and check means:
         initial value was '' or 'h'
        """
        self.SID = int(SID)
        self.sent = token[0]
        self.WID = int(token[1])
        self.token = token[2]
        self.GS_head = self.get_value(token[3])
        self.head = self.get_value(token[4])
        self.new_head = token[5]
        self.check = self.check_tok(token[6])
        self.gram = token[7]
        self.lemma = token[8]
        self.lemma = self.lemma

    def get_value(self, value):
        if value == '' or value == 'h': return 00
        else: return int(value)

    def check_tok(self, value):
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

    def get_pairs(self, childHeadfeauture=None):
        self.pairs = []
        self.true_pairs = []
        for self.head in self.sentence:
            for self.child in self.sentence2:
                if childHeadfeauture == 'GS_head':
                    if self.head.WID == self.child.GS_head:
                        self.pairs.append([self.head, self.child])
                elif childHeadfeauture == 'head':
                    if self.head.WID == self.child.head and (self.child.head != '' or self.child.head != 00):
                        self.pairs.append([self.head, self.child])
                elif childHeadfeauture == 'all':
                    if self.head.WID == self.child.head and (self.child.head != '' or self.child.head != 00):
                        self.pairs.append([self.head, self.child])
                    if self.head.WID == self.child.GS_head:
                        true_child = self.child
                        true_child.check = True
                        true_child.head = true_child.GS_head
                        pair = [self.head, true_child]
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
        self.dep_pairs_by_POS = []
        self.POSfeature = POSfeature
        self.POSasWhat = POSasWhat
        for self.head in self.sentence:
            for self.child in self.sentence2:
                if self.POSasWhat == 'head':
                    if self.POSfeature in self.head.gram:
                        if self.child.head == self.head.WID:
                                self.dep_pairs_by_POS.append([self.head, self.child])
                elif self.POSasWhat == 'child':
                        if self.POSfeature in self.child.gram:
                            if self.child.head == self.head.WID:
                                self.dep_pairs_by_POS.append([self.head, self.child])
                elif self.POSasWhat == 'GS_head':
                    if self.POSfeature in self.head.gram:
                        if self.head.WID == self.child.GS_head:
                           self.dep_pairs_by_POS.append([self.head, self.child])
                elif self.POSasWhat == 'GS_child':
                    if self.POSfeature in self.child.gram:
                        if self.head.WID == self.child.GS_head:
                            self.dep_pairs_by_POS.append([self.head, self.child])
        return self.dep_pairs_by_POS

    def position(self, head, child):
        """Вернуть позицию ребёнка по отношению к главному слову
        """
        if head.WID < child.WID:  return 'pre-pos'
        elif head.WID > child.WID: return 'post-pos'


class CommonRule(object):
    """Класс для описания связи <head - child> из данных типа <SentenceRule>."""
    def __init__(self, dep_pair, tags=None):
        self.dep_pair = dep_pair
        self.head, self.child = self.dep_pair
        self.id = '%s.%s.%s' % (self.head.SID, self.head.WID, self.child.WID)
        self.headToken = self.head.token
        self.childToken = self.child.token
        self.sentence = self.head.sent
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
        self.sentence_length_in_words = len(self.sentence)
        self.sentence_length_in_chars = self.sentence_len_chars()
        self.child_has_children = self.check_for_children(self.child)
        self.head_has_other_children = self.check_for_children(self.head)
        self.head_is_smth_else_child = self.check_for_head(self.head)
        self.head_is_root = self.check_root(self.head)
        self.child_is_root = self.check_root(self.child)
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

    def check_root(self, what):
        if what.head == 0 or what.head == '0':
            return True
        else: return False

    def check_is_void(self, what):
        if what == 00: return True
        else: return False

    def position(self, head, child):
        if head.WID < child.WID:  return 'pre-pos'
        elif head.WID > child.WID: return 'post-pos'

    def words_before(self, what=None, bit=None):
        if what is not None:
            condition = lambda  word, what : word.head == what.WID \
                                         and (word.WID < self.head.WID and word.WID < self.child.WID
                                                             and word.head != '' and word.head != 00)
        else:
            condition = lambda  word, what : (word.WID < self.head.WID and word.WID < self.child.WID
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
            condition = lambda  word, what : ((word.WID < self.head.WID and word.WID > self.child.WID)
                                            or (word.WID > self.head.WID and word.WID < self.child.WID)
                                            and (word.head == what.WID and word.head != '' and word.head != 00))
        else:
            condition = lambda  word, what : (((word.WID < self.head.WID and word.WID > self.child.WID)
                                           or (word.WID > self.head.WID and word.WID < self.child.WID))
                                                and word.head != '' and word.head != 00)
        return self.words_at_positions(what=what, bit=bit, condition=condition)

    def words_dependencies(self, what=None, bit=None, how=None):
        condition = lambda  word: word.head == what.WID
        if how == 'between':
            words = self.words_between(what=what)
        elif how == 'after':
            words = self.words_after(what=what)
        elif how == 'before':
            words = self.words_before(what=what)
        words = [word for word in words if condition(word)]
        if words != [] and bit is True: return True
        elif words != [] and bit is None: return words
        else: return False

    def sentence_len_chars(self):
        condition  = lambda  x: 1 > 0
        return self.len_chars(condition=condition)

    def len_chars(self, condition=None):
        return sum([len(word.token) for word in self.sentence if
                       condition(word)])

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

    def get_pos(self, token, *args):
        for arg in args:
            if arg in token.gram: return True
        else: return False


def read_file(filename, SyntAutom=None, tags=None):
    with open(filename, 'rb') as infile:
        csv_reader = csv.DictReader(infile, delimiter=';')
        cur_sent_id = None
        results = dict()
        tags = set()
        links = set()
        for row in csv_reader:
            if not row['SID']:
                    row['SID'] = cur_sent_id
            else:
                    cur_sent_id = row['SID']
            if cur_sent_id not in results:
                    results[cur_sent_id] = []
            if SyntAutom is not None:
                    if not row['SA/Check']: check = 1
                    else: check = row['SA/Check']
                    results[cur_sent_id].append([[word.decode('cp1251') for word in row['Sent'].split()], row['WID'],
                                                 row['Token'].decode('cp1251'),
                                             row['GS/Head'], row['SA/Head'], row['SA/New Head'],
                                             check, row['SA/Gramm'], row['Lemma'].decode('cp1251')])
            if tags is True:
                for tag in row['SA/Gramm'].split(','):
                    if tag not in tags and tag != '':
                        tags.update(tags)
                if row['Type'] not in links:
                    links.update(row['Type'])

    if len(tags) > 0: return results, tags, links
    else: return results


def build_features(in_fn):

    results, tags, links = read_file(in_fn, tags=True)
    token_sentences = [[Token(feat, sent) for feat in results[sent]] for n, sent in enumerate(results)]

    genius_move = [CommonRule(dep_pair, tags)
                   for sentence in token_sentences
                   for dep_pair in SentenceRule(sentence).get_pairs('all')]

    return genius_move


def save_extracted_feats(outfn, outdir, results):
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


def _profile_it(in_fn, featsdir, docname='out_feats_profile.csv'):
    import cProfile
    import pstats
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    cProfile.run('main()', 'cProfile.tmp')
    c = pstats.Stats('cProfile.tmp')
    c.sort_stats('ncalls').print_stats(50)
    with PyCallGraph(output=GraphvizOutput()):
            build_features(in_fn)


if __name__ == '__main__':
    in_fn = sys.argv[1]
    featsdir = sys.argv[2]
    docname = 'out_feats_15.csv'
    results = build_features(in_fn)
    save_extracted_feats(outdir=featsdir, outfn=docname, results=results)







