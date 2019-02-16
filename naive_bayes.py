import random


class NaiveBayes:

    def __init__(self):
        self.freq_table = {}
        self.likelihood_table = {}
        self.all_points = 0

    def to_frequency_table(self, data):
        self.all_points = len(data)
        for data_point in data:
            if data_point[0] in self.freq_table.keys():
                self.freq_table[data_point[0]][data_point[1]] += 1
            else:
                self.freq_table.setdefault(data_point[0], [0, 0])
                self.freq_table[data_point[0]][data_point[1]] += 1

    def to_likelihood_table(self):
        first_cls_prob = sum(i[0] for i in self.freq_table.values()) / self.all_points
        snd_cls_prob = sum(i[1] for i in self.freq_table.values()) / self.all_points
        point_prob = {k: sum(v) / self.all_points for k, v in self.freq_table.items()}
        likelihood = [first_cls_prob, snd_cls_prob] + list(point_prob.values())
        names = [0, 1] + list(self.freq_table.keys())
        self.likelihood_table = {k: v for k, v in zip(names, likelihood)}

    def fit(self, data):
        self.to_frequency_table(data)
        self.to_likelihood_table()

    def conditional_prob(self, event, condition):
        # prob(cond | event) * prob(event) / prob(cond)
        prob_of_event = self.likelihood_table[event]
        prob_of_condition = self.likelihood_table[condition]
        prob_of_cond_given_event = self.freq_table[condition][event] / sum(i[event] for i in self.freq_table.values())
        return (prob_of_cond_given_event * prob_of_event) / prob_of_condition

    def predict(self, data):
        prob_of_fst_cls = self.conditional_prob(0, data)
        prob_of_snd_cls = self.conditional_prob(1, data)
        if prob_of_fst_cls > prob_of_snd_cls:
            return 0
        elif prob_of_fst_cls < prob_of_snd_cls:
            return 1
        else:
            return random.choice([0, 1])


def main():
    data = [['Sunny', 0], ['Overcast', 1], ['Rainy', 1], ['Sunny', 1], ['Sunny', 1],
            ['Overcast', 1], ['Rainy', 0], ['Rainy', 0], ['Sunny', 1], ['Rainy', 1],
            ['Sunny', 0], ['Overcast', 1], ['Overcast', 1], ['Rainy', 0]]
    nb = NaiveBayes()
    nb.fit(data)
    print(nb.freq_table)
    print(nb.likelihood_table)
    print(nb.conditional_prob(1, 'Sunny'))
    print(nb.predict('Rainy'))


if __name__ == '__main__':
    main()
