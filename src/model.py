import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y

word2vec_path = '../data/content_data/word2vec_wv.model'
doc2vec_path = '../data/content_data/doc2vec.npy'
doc_topics_path = '../data/content_data/doc_topics.npy'


class MODEL(object):
    def __init__(self, args):
        self.params = []
        self._build_inputs(args)
        self._build_model(args)
        self._build_train(args)

    def _build_inputs(self, args):
        with tf.name_scope('input'):
            self.news_title = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='news_title')
            self.clicked_titles = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_titles')
            self.active_time = tf.placeholder(
                dtype=tf.float32, shape=[None, 20], name='active_time')
            self.labels = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name='labels')
            self.time_label = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name='time_label')
            self.news_category = tf.placeholder(
                dtype=tf.int32, shape=[None, 3], name='category')
            self.user_city = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='city')
            self.user_region = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='region')
            self.clicked_category = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, 3], name='clicked_category')
            self.news_age = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='news_age')
            self.news_id = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='news_id')
            self.news_len = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='news_len')
            self.clicked_news_id = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, 1], name='clicked_news_id')
            self.clicked_news_len = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, 1], name='clicked_news_len')
            self.is_train = tf.placeholder(dtype=bool, shape=None, name='is_train')
            self.weights = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='weights')

    def _build_model(self, args):
        with tf.name_scope('embeddings', ):
            word_embs = KeyedVectors.load_word2vec_format(word2vec_path)
            doc_emb = np.load(doc2vec_path).tolist()
            topic_emb = np.load(doc_topics_path).tolist()

            zero_word_embedding = tf.zeros(shape=[1, 100], dtype=tf.float32)
            word_embedding = tf.Variable(word_embs.vectors, dtype=tf.float32, trainable=True, name='word_embedding')
            self.word_embeddings = tf.concat([zero_word_embedding, word_embedding], axis=0)

            zero_doc_embedding = tf.zeros(shape=[1, 100], dtype=tf.float32)
            doc_embedding = tf.Variable(doc_emb, dtype=tf.float32, name='doc_embedding', trainable=False)
            self.doc_embeddings = tf.concat([zero_doc_embedding, doc_embedding], axis=0)

            zero_category_embedding = tf.zeros(shape=[1, 100], dtype=tf.float32)
            category_embedding = tf.Variable(tf.random_normal((260, 100), stddev=1), trainable=True,
                                             name='category_embedding')
            self.category_embeddings = tf.concat([zero_category_embedding, category_embedding], axis=0)

            self.topic_embeddings = tf.Variable(topic_emb, dtype=tf.float32, trainable=False, name='topic_embedding')


            self.city_embeddings = tf.Variable(tf.random_normal((5050, 50), stddev=1), trainable=True,
                                               name='city_embedding')
            self.region_embeddings = tf.Variable(tf.random_normal((1100, 50), stddev=1), trainable=True,
                                                 name='region_embedding')
            self.age_embeddings = tf.Variable(tf.random_uniform(shape=[90, 100], minval=-1, maxval=1), trainable=True,
                                              name='age_embedding')

            zero_len_embedding = tf.zeros(shape=[1, 50], dtype=tf.float32)
            len_embedding = tf.Variable(tf.random_uniform(shape=[1000, 50], minval=-1, maxval=1), trainable=True,
                                        name='len_embedding')
            self.len_embeddings = tf.concat([zero_len_embedding, len_embedding], axis=0)

        explicit_embedding = self._explicit_encoder(args, self.news_title, self.news_category)
        implicit_embedding = self._implicit_encoder(args, self.news_id, self.news_len)
        # news_embedding=self._decay_unit(explicit_embedding,implicit_embedding)
        # news_embedding=tf.add(explicit_embedding,implicit_embedding)
        news_embedding_p = self._attention_unit(args, explicit_embedding, implicit_embedding, tag='p')
        news_embedding_t = self._attention_unit(args, explicit_embedding, implicit_embedding, tag='t')
        news_embedding_p = self._decay_unit(args, news_embedding_p)
        # news_embedding_t = self._decay_unit(args, news_embedding_t)

        user_embedding = self._user_encoder(args)

        self.news_embedding = news_embedding_p
        self.user_embedding = user_embedding

        self.unnormized_p = tf.reshape(tf.diag_part(tf.matmul(news_embedding_p, user_embedding, transpose_b=True)),
                                       shape=[-1, 1])
        self.unnormized_p = self.batch_normal(self.unnormized_p, args)
        self.p = tf.sigmoid(self.unnormized_p)

        with tf.variable_scope('w_classification', reuse=tf.AUTO_REUSE):
            w_cla = tf.get_variable(name='w_cla', shape=[400, 20], dtype=tf.float32, trainable=True)
        self.user_news = tf.concat([user_embedding, news_embedding_t], axis=-1)
        self.time_score = tf.matmul(self.user_news, w_cla)
        self.time_score = tf.nn.softmax(self.time_score, axis=-1)
        self.time_pre = tf.argmax(self.time_score, 1)
        self.time_true = tf.argmax(self.active_time, 1)

    def _explicit_encoder(self, args, news_title, news_category):
        # news_categorys=self.id2categorys[self.news_id]
        title = tf.nn.embedding_lookup(self.word_embeddings, news_title)
        category = tf.nn.embedding_lookup(self.category_embeddings, news_category)
        mean_category = tf.reduce_mean(category, axis=1)
        _mean_category = tf.reshape(mean_category, [-1, 100])
        # title=tf.reshape(title,[args.batch_size,args.max_title_length,args.word_dim])
        # categorys=tf.nn.embedding_lookup(self.category_embeddings,news_categorys)
        title_emb = []
        for filter_size in args.filter_sizes:
            conv = tf.layers.conv1d(inputs=title, filters=args.n_filters, kernel_size=filter_size)
            conv = self.batch_normal(conv, args)
            pool = tf.layers.max_pooling1d(inputs=conv, pool_size=args.max_title_length - filter_size + 1, strides=1)
            title_emb.append(pool)
        title_emb = tf.concat(title_emb, axis=1)
        title_emb = tf.reshape(title_emb, [-1, args.n_filters * len(args.filter_sizes)])
        # category_emb=tf.layers.dense(inputs=category,units=100, activation=tf.nn.relu)
        # d2 = tf.layers.dropout(d2, rate=args.dropout_rate, training=self.is_train)
        # explicit_emb=tf.concat([title_emb,category_emb])
        concat_emb = tf.concat([title_emb, _mean_category], axis=1)
        # explicit_emb=title_emb
        explicit_emb = tf.layers.dense(inputs=concat_emb, units=200)
        explicit_emb = tf.layers.dropout(inputs=explicit_emb, rate=args.dropout_rate, training=self.is_train)
        explicit_emb = tf.tanh(self.batch_normal(explicit_emb, args))
        return tf.reshape(explicit_emb, [-1, 200])

    def _implicit_encoder(self, args, news_id, news_len):
        news_topic = tf.nn.embedding_lookup(self.topic_embeddings, news_id)
        news_text = tf.nn.embedding_lookup(self.doc_embeddings, news_id)
        len_emb = tf.nn.embedding_lookup(self.len_embeddings, news_len)
        len_emb = tf.layers.dense(inputs=len_emb, units=100)
        len_emb = tf.layers.dropout(inputs=len_emb, rate=args.dropout_rate, training=self.is_train)
        len_emb = tf.tanh(self.batch_normal(len_emb, args))
        # implicit_emb=tf.concat([news_topic,len_emb],axis=-1)
        implicit_emb = tf.concat([news_topic, news_text, len_emb], axis=-1)
        implicit_emb = tf.layers.dense(inputs=implicit_emb, units=200)
        implicit_emb = tf.layers.dropout(inputs=implicit_emb, rate=args.dropout_rate, training=self.is_train)
        implicit_emb = tf.tanh(self.batch_normal(implicit_emb, args))
        return tf.reshape(implicit_emb, [-1, 200])

    def _attention_unit(self, args, explicit_emb, implicit_emb, tag):
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            if tag == 'p':
                attention_w = tf.get_variable(name='attention_p_w', shape=[200, 200], dtype=tf.float32, trainable=True)
                attention_b = tf.get_variable(name='attention_p_b', dtype=tf.float32, shape=[200, 1], trainable=True)
                attention_q = tf.get_variable(name='attention_p_q', dtype=tf.float32, shape=[1, 200], trainable=True)
            elif tag == 't':
                attention_w = tf.get_variable(name='attention_t_w', shape=[200, 200], dtype=tf.float32, trainable=True)
                attention_b = tf.get_variable(name='attention_t_b', dtype=tf.float32, shape=[200, 1], trainable=True)
                attention_q = tf.get_variable(name='attention_t_q', dtype=tf.float32, shape=[1, 200], trainable=True)
            elif tag == 'u':
                attention_w = tf.get_variable(name='attention_u_w', shape=[200, 200], dtype=tf.float32, trainable=True)
                attention_b = tf.get_variable(name='attention_u_b', dtype=tf.float32, shape=[200, 1], trainable=True)
                attention_q = tf.get_variable(name='attention_u_q', dtype=tf.float32, shape=[1, 200], trainable=True)

            # attention_w = tf.layers.dropout(inputs=attention_w, rate=args.dropout_rate, training=self.is_train)
            # attention_b = tf.layers.dropout(inputs=attention_b, rate=args.dropout_rate, training=self.is_train)
            # attention_q = tf.layers.dropout(inputs=attention_q, rate=args.dropout_rate, training=self.is_train)

            att1 = tf.reshape(tf.matmul(attention_q, tf.tanh(tf.matmul(attention_w, explicit_emb, transpose_b=True) + attention_b)), shape=[-1, 1])
            att2 = tf.reshape(tf.matmul(attention_q, tf.tanh(tf.matmul(attention_w, implicit_emb, transpose_b=True) + attention_b)), shape=[-1, 1])
            att = tf.reshape(tf.concat([att1, att2], axis=-1), shape=[-1, 2])
            norminized_att = tf.nn.softmax(att, axis=1)

            news_emb = tf.multiply(explicit_emb, tf.slice(norminized_att, [0, 0], [-1, 1])) + tf.multiply(implicit_emb, tf.slice(norminized_att, [0, 1], [-1, 1]))
            return news_emb


    def _decay_unit(self, args, news_embedding):

        age_emb = tf.nn.embedding_lookup(self.age_embeddings, self.news_age)
        age_emb = tf.reshape(tf.layers.dense(inputs=age_emb, units=200), [-1, 200])
        age_emb = tf.tanh(self.batch_normal(age_emb, args))
        with tf.variable_scope('decay', reuse=tf.AUTO_REUSE):
            decay_w = tf.get_variable(dtype=tf.float32, shape=[200, 200], name='decay_w', trainable=True)
            decay_b = tf.get_variable(dtype=tf.float32, shape=[1, 200], name='decay_b', trainable=True)
        alpha_d = tf.reshape(tf.tanh(tf.add(tf.matmul(age_emb, decay_w), decay_b)), shape=[-1, 200])
        decay_news = tf.multiply(alpha_d, news_embedding)
        # decay_news=tf.add(alpha_d, news_embedding)
        return decay_news

    def _user_encoder(self, args):
        # clicked_history=[]
        clicked_titles = tf.reshape(self.clicked_titles, shape=[-1, args.max_title_length])
        clicked_category = tf.reshape(self.clicked_category, shape=[-1, 3])
        clicked_news_id = tf.reshape(self.clicked_news_id, shape=[-1, 1])
        clicked_news_len = tf.reshape(self.clicked_news_len, shape=[-1, 1])
        # news_emb=tf.concat([self._explicit_encoder(args,news),self._implicit_encoder(args,news)])
        # news_emb=tf.add_n(self._explicit_encoder(args,title),self._implicit_encoder(args,title))
        exp_emb = self._explicit_encoder(args, clicked_titles, clicked_category)
        # exp_emb=tf.reshape(exp_emb,shape=[-1,args.max_click_history,200])
        imp_emb = self._implicit_encoder(args, clicked_news_id, clicked_news_len)
        # imp_emb=tf.reshape(imp_emb,shape=[-1,args.max_click_history,200])
        news_emb = self._attention_unit(args, exp_emb, imp_emb, 'u')
        news_emb = tf.reshape(news_emb, shape=[-1, args.max_click_history, 200])

        with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
            self_w = tf.get_variable(name='self_w', shape=[200, 200], dtype=tf.float32, trainable=True)
            self_b = tf.get_variable(name='self_b', shape=[200, 1], dtype=tf.float32, trainable=True)
            self_q = tf.get_variable(name='self_1', shape=[1, 200], dtype=tf.float32, trainable=True)
            alpha = []
            for i in range(args.max_click_history):
                alpha_i = tf.matmul(self_q, tf.matmul(self_w, tf.reshape(news_emb[:, i, :], shape=[-1, 200]),
                                                      transpose_b=True) + self_b)
                alpha.append(alpha_i)
            alpha = tf.nn.softmax(tf.transpose(alpha), axis=-1)
        clicked_sum = tf.reduce_sum(tf.multiply(news_emb, tf.reshape(alpha, shape=[-1, args.max_click_history, 1])),
                                    axis=1)
        clicked_emb = tf.reshape(clicked_sum, shape=[-1, 200])

        return clicked_emb

    def _build_train(self, args):
        with tf.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.unnormized_p))

            idx = tf.reshape(tf.where(self.time_label > 0)[:, 0], [-1, 1])

            active_time = tf.gather_nd(self.active_time, idx)
            time_score = tf.gather_nd(self.time_score, idx)
            weight = tf.reshape(tf.gather_nd(self.weights, idx), [-1, 1])

            count = tf.reduce_sum(self.active_time, 0)
            count_reci = tf.reciprocal(count)

            self.classify_loss = -tf.reduce_mean(
                tf.reshape(tf.reduce_sum(self.active_time * tf.log(self.time_score + 0.0000001), axis=-1), [-1, 1]))

            tv = tf.trainable_variables()
            l2_loss_list = []
            for v in tv:
                if '_embedding' not in v.name:
                    l2_loss_list.append(tf.nn.l2_loss(v))
            self.l2_loss = tf.reduce_sum(l2_loss_list)

            self.loss = self.base_loss + args.time_weight * self.classify_loss + args.l2_weight * self.l2_loss

            self.optimizer = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run(self.optimizer, feed_dict)

    def infer(self, sess, feed_dict):
        scores = sess.run([self.p], feed_dict)
        return scores

    def infer_news_embedding(self, sess, feed_dict):
        news_embedding = sess.run([self.news_embedding], feed_dict)
        return news_embedding

    def infer_user_embedding(self, sess, feed_dict):
        user_embedding = sess.run([self.user_embedding], feed_dict)
        return user_embedding

    def test(self, sess, feed_dict):
        loss, time_loss, labels, scores, time_pre, time_true, time_label = sess.run(
            [self.loss, self.classify_loss, self.labels, self.p, self.time_pre, self.time_true, self.time_label],
            feed_dict)
        # auc=roc_auc_score(y_true=labels,y_score=scores)
        return loss, labels, scores, time_loss, time_pre, time_true, time_label

    def get_attention(self, sess, feed_dict):
        att_p, att_t, att_u = sess.run([self.att_p, self.att_t, self.att_u], feed_dict)
        return att_p, att_t, att_u

    def eval(self, args, labels, scores, time_true, time_pre):
        labels = np.array(labels).reshape(-1, 1)
        scores = np.array(scores).reshape(-1, 1)
        time_true = np.array(time_true).reshape(-1, 1)
        time_pre = np.array(time_pre).reshape(-1, 1)
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except:
            auc = 0
        try:
            f1 = f1_score(y_true=labels, y_pred=[int(item > args.threshold) for item in scores])
        except:
            f1 = 0
        # ndcg_tuple=(scores,labels)
        try:
            pre = precision_score(y_true=labels, y_pred=[int(item > args.threshold) for item in scores])
        except:
            pre = 0
        try:
            recall = recall_score(y_true=labels, y_pred=[int(item > args.threshold) for item in scores])
        except:
            recall = 0
        try:
            time_f1 = f1_score(y_true=time_true, y_pred=time_pre, average='micro')
        except:
            time_f1 = 0

        return auc, f1, pre, recall, time_f1

    def batch_normal(self, inputs, args):
        if args.is_train:
            off = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
            mean, var = tf.nn.moments(inputs, [0])
            epsilon = 0.001
            return tf.nn.batch_normalization(inputs, mean, var, off, scale, epsilon)
        return inputs

    def calc_dcg(self, sorted_vec, at):
        import math
        ranking = [t[1] for t in sorted_vec[0: at]]
        dcg_ = sum([(2 ** r - 1) / math.log(i + 2, 2) for i, r in enumerate(ranking)])
        return dcg_

    def calc_ndcg(self, vec, at):
        sorted_vec = sorted(vec, key=lambda t: t[1], reverse=True)
        ideal_dcg = self.calc_dcg(sorted_vec, at)
        sorted_vec = sorted(vec, key=lambda t: t[0], reverse=True)
        cur_dcg = self.calc_dcg(sorted_vec, at)
        if ideal_dcg == 0:
            return 0
        else:
            return cur_dcg / ideal_dcg

    def dcg_score(self, y_true, y_score, k):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gain = 2 ** y_true - 1
        # print(gain)
        discounts = np.log2(np.arange(len(y_true)) + 2)
        # print(discounts)
        return np.sum(gain / discounts)

    def ndcg_score(self, y_true, y_score, k):
        y_score, y_true = check_X_y(y_score, y_true)

        # Make sure we use all the labels (max between the length and the higher
        # number in the array)
        lb = LabelBinarizer()
        lb.fit(np.arange(max(np.max(y_true) + 1, len(y_true))))
        binarized_y_true = lb.transform(y_true)
        print(binarized_y_true)
        if binarized_y_true.shape != y_score.shape:
            raise ValueError("y_true and y_score have different value ranges")

        scores = []

        # Iterate over each y_value_true and compute the DCG score
        for y_value_true, y_value_score in zip(binarized_y_true, y_score):
            actual = dcg_score(y_value_true, y_value_score, k)
            best = dcg_score(y_value_true, y_value_true, k)
            # print(best)
            scores.append(actual / best)
        return np.mean(scores)

