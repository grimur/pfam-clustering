from __future__ import print_function

from multiprocessing.pool import ThreadPool
from numpy import ma
import numpy
import pandas

CLOSE_TO_INF = 1e-200

pool = ThreadPool(processes=8)


class Multinomial(object):
    def __init__(self):
        self.parameters = {}
        self.labels_verified = False

    def p(self, value):

        if value in self.parameters:
            probability = self.parameters[value]
        else:
            probability = CLOSE_TO_INF

        if probability == 0.0:
            probability = CLOSE_TO_INF

        return probability

    def verify_labels(self, labels):
        labels_to_add = []
        for label in labels:
            if label not in self.parameters.keys():
                labels_to_add.append(label)
        if len(labels_to_add) > 0:
            new_label_probability = 1.0 / (len(self.parameters.keys()) + len(labels_to_add))
            old_label_scaling_factor = float(len(self.parameters.keys())) / len(labels_to_add)
            for label in self.parameters.keys():
                self.parameters[label] *= old_label_scaling_factor
            for label in labels_to_add:
                self.parameters[label] = new_label_probability

    def verify_labels_list(self, labels):
        if not self.labels_verified:
            labels = set(labels)
            self.verify_labels(labels)
            self.labels_verified = True


class Mixture(object):
    def __init__(self):
        self.words = []
        self.distributions = []

    def initialise_mixture(self, count):
        # self.distributions = []
        # for c in xrange(count):
        #     self.distributions.append(Multinomial())
        self.distributions = [Multinomial() for x in xrange(count)]

    def p(self, point):
        # probability that p is drawn from the distribution.
        # Probably highly dependent on the covariance matrix?
        # Assume independence for now, in accordance with pdf pg. 25
        # final probability is the product of component probabilities.
        p = 1.0
        for distribution, dimension in zip(self.distributions, point):
            p *= distribution.p(dimension)
        return p

    def p_log(self, point):
        p = 0
        for distribution, dimension in zip(self.distributions, point):
            p_dist = distribution.p(dimension)
            if p_dist == 0:
                p_dist = CLOSE_TO_INF
            # p += numpy.log(distribution.p(dimension))
            p += numpy.log(p_dist)
        return p

    def update_new(self, membership, data):
        membership_sum = sum(membership)
        jobs = []
        for dimension_index, dimension in enumerate(self.distributions):
            dimension_data = data[:, dimension_index]
            dimension.verify_labels_list(dimension_data)
            new_parameters = update_parameters(dimension.parameters, dimension_data, membership, membership_sum)
            dimension.parameters = new_parameters
            # print(dimension.parameters == new_parameters)
        #    p = pool.apply_async(update_parameters, args=(dimension.parameters, dimension_data, membership, membership_sum))
        #    jobs.append(p)

        #for p, dimension in zip(jobs, self.distributions):
        #    new_parameters = p.get()
        #    dimension.parameters = new_parameters

    def update(self, membership, data):
        for dimension_index, dimension in enumerate(self.distributions):
            dimension_data = data[:, dimension_index]
            dimension.verify_labels(set(dimension_data))
            for label, old_probability in dimension.parameters.items():
                dimension_binary_vector = dimension_data == label
                new_probability = sum([x*y for x, y in zip(dimension_binary_vector, membership)]) / sum(membership)
                dimension.parameters[label] = new_probability


def update_parameters(old_parameters, dimension_data, membership, membership_sum):
    new_parameters = {}
    for label, old_probability in old_parameters.items():
        dimension_binary_vector = dimension_data == label
        new_probability = sum([x * y for x, y in zip(dimension_binary_vector, membership)]) / membership_sum
        new_parameters[label] = new_probability
    return new_parameters
    


class MixtureModel(object):
    def __init__(self, n_components):
        # each distribution
        self.components = [Mixture() for x in xrange(n_components)]
        # self.components = []
        # for x in xrange(n_components):
        #     self.components.append(Mixture())

        # the probability of each distribution in the generative model
        # prior probability - equal
        self.weights = [1.0 / n_components for x in xrange(n_components)]
        # self.weights = numpy.random.uniform(0, 1, n_components)

        # termination conditions
        self.max_iterations = 10000
        self.epsilon = 1e-5

        self._termination_counter = 0
        self._last_accuracy = 0

        self.current_probabilities = None

    def fit(self, data, max_iter=None, epsilon=None, verbal=False):
        if epsilon is not None:
            self.epsilon = epsilon
        if max_iter is not None:
            self.max_iterations = max_iter

        if type(data) == pandas.DataFrame:
            self.data = data.values
            self.raw_data = data
        else:
            self.data = data
            self.raw_data = data

        # We need to track:
        # - the membership of each data point (num_samples x num_clusters)
        # self.membership = numpy.ones((len(data), len(self.components))) * 1.0 / len(self.components)
        self.membership = numpy.random.uniform(0, 1, size=(len(data), len(self.components)))

        # number of discrete words / dimensions
        # words = set(numpy.ravel(data))
        dimensions = data.shape[1]
        [x.initialise_mixture(dimensions) for x in self.components]

        num_clusters = len(self.components)
        num_samples = len(data)

        if self.current_probabilities is None:
            self.current_probabilities = numpy.zeros((num_samples, num_clusters)) / num_clusters

        verbal = True

        while not self.check_termination(verbal):
            self.m_step()
            self.e_step()

    def accuracy_new(self):
        num_samples, num_dimensions = self.data.shape
        num_clusters = len(self.components)

        p_k = ma.log(numpy.array(self.weights * num_samples).reshape((num_samples, num_clusters)))
        p_k_xi = self.current_probabilities
        q_ik = self.membership

        # q_ik[q_ik == 0] = 1
        # p_k_xi[p_k_xi == 0] = 1

        # l_q_ik = numpy.log(q_ik)
        # l_p_k_xi = numpy.log(p_k_xi)

        # Use masked arrays to remove nan values
        l_q_ik = ma.log(q_ik)
        l_p_k_xi = ma.log(p_k_xi)

        accuracy = numpy.sum(q_ik * (p_k.filled(0) + l_p_k_xi.filled(0) - l_q_ik.filled(0)))

        return accuracy

    def accuracy(self):
        num_samples, num_dimensions = self.data.shape
        num_clusters = len(self.components)

        accuracy = 0

        for k in xrange(num_clusters):
            p_k = numpy.log(self.weights[k])
            for i in xrange(num_samples):
                q_ik = self.membership[i, k]
                p_k_xi = self.components[k].p(self.data[i, :])
                # p_k_xi = self.current_probabilities[i, k]

                if q_ik == 0:
                    l_q_ik = 0
                else:
                    l_q_ik = numpy.log(q_ik)

                if p_k_xi == 0:
                    l_p_k_xi = 0
                else:
                    l_p_k_xi = numpy.log(p_k_xi)

                accuracy += q_ik * (p_k + l_p_k_xi - l_q_ik)

        return accuracy

    def improvement(self):
        accuracy = self.accuracy_new()
        delta = accuracy - self._last_accuracy
        self._last_accuracy = accuracy
        return numpy.abs(delta)

    def check_termination(self, verbal=True):
        improvement = self.improvement()

        if verbal:
            print('Iter %s, acc %s, delta %s' % (self._termination_counter, self._last_accuracy, improvement))

        if self._termination_counter > self.max_iterations:
            return True
        else:
            self._termination_counter += 1
        if improvement < self.epsilon:
            return True
        return False

    def e_step(self):
        # The argument is updated in place. To compare two functions, create
        # two dummy matrices and pass them as arguments, assigning the old method
        # to self.membership.

        # to_be_updated_2 = numpy.zeros_like(self.membership)
        # self.do_e_step_new(to_be_updated_2)
        self.do_e_step_new(self.membership)

    def do_e_step_new(self, output):
        jobs = []
        for i, point in enumerate(self.data):
            p_vector, vector = e_step_inner_loop(self.components, self.weights, point)
            # self.membership[i, :] = vector
            output[i, :] = vector
            self.current_probabilities[i, :] = p_vector

        #     p = pool.apply_async(e_step_inner_loop, args=(self.components, self.weights, point))
        #     jobs.append(p)

        # for i, p in enumerate(jobs):
        #     p_vector, vector = p.get()
        #     self.membership[i, :] = vector
        #     self.current_probabilities[i, :] = p_vector

    def do_e_step(self, output):
        # update the probabilitiy matrix for sample/cluster
        for i, point in enumerate(self.data):
            sum_of_weights = numpy.sum([c.p(point) * x for x, c in zip(self.weights, self.components)])
            for j, cluster in enumerate(self.components):
                p_point = cluster.p(point)
                p_w = self.weights[j]
                w = p_point * p_w / sum_of_weights
                # self.membership[i, j] = w
                output[i, j] = w

    def m_step(self):
        # update the weight vector
        N = len(self.data)
        for i in xrange(len(self.weights)):
            self.weights[i] = numpy.sum(self.membership[:, i]) / N

        # update the mixtures
        # See clustering pdf pg 25
        for j in xrange(len(self.components)):
            membership_probabilities = self.membership[:, j]
            self.components[j].update_new(membership_probabilities, self.data)

    def predict_proba(self, data):
        predictions = []
        for v in data:
            res = [w * c.p(v) for w, c in zip(self.weights, self.components)]
            print(res)
            res_norm = numpy.sum(res)
            if res_norm == 0:
                print('Underflow warning')
            predictions.append([x/res_norm for x in res])
        return predictions

    def predict(self, data):
        predictions = self.predict_proba(data)
        res = [numpy.argmax(x) for x in predictions]
        return res

    def predict_with_proba(self, data):
        predictions = self.predict_proba(data)
        res = [numpy.argmax(x) for x in predictions]
        return zip(res, [p[x] for p, x in zip(predictions, res)])

    def predict_log(self, data):
        predictions = []
        for v in data:
            res = [c.p_log(v) for c in self.components]
            predictions.append(res)

        max_pred = numpy.max(predictions)
        predictions = numpy.array(predictions) - max_pred

        return predictions

    def predict_proba(self, data):
        log_predictions = self.predict_log(data)
        predictions = numpy.exp(log_predictions)

        predictions = [w * x for w, x in zip(self.weights, predictions)]

        # normalise
        predictions = numpy.array([x / numpy.sum(x) for x in predictions])

        return predictions


def e_step_inner_loop(components, weights, point):
    c_p = [c.p(point) for c in components]
    sum_of_weights = sum([w * p for w, p in zip(weights, c_p)])
    vector = [w * p / sum_of_weights for w, p in zip(weights, c_p)]
    return c_p, vector


if __name__ == '__main__':
    print('init')
    mm = MixtureModel(2)
    print('define data')
    data = numpy.array(
        [
            [1, 1, 1, 1, 0],
            [2, 1, 2, 0, 0],
            [1, 2, 1, 0, 0],
            [3, 2, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 4, 1],
            [0, 0, 1, 1, 2],
            [3, 2, 1, 1, 1]
        ]
    )
    data3 = numpy.array(
        [
            [0, 2],
            [0, 1],
            [1, 0],
            [2, 0],
            # [3, 1]
        ]
    )
    print('fit')
    mm.fit(data)

    print('predict')
    data2 = numpy.array(
        [
            [1, 1]
        ]
    )
    res = mm.predict_proba(data)
    print(res)
