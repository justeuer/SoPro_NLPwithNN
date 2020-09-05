from collections import Counter
from matplotlib import pyplot as plt
import nltk
from typing import List


class LevenshteinDistance(object):

    def __init__(self,
                 true: List[str],
                 pred: List[str],
                 upper_bound=5):
        self.true = true
        self.pred = pred
        self.upper_bound = upper_bound
        self.distances = sorted([self._levenshtein(t, p) for t, p in zip(true, pred)], reverse=True)
        self.percentiles = self._percentiles()

    def _levenshtein(self, t: str, p: str):
        distance = nltk.edit_distance(t, p)
        return min(distance, self.upper_bound)

    def _percentiles(self):
        prev = 0
        data = Counter(self.distances)
        percentiles = {}

        # add up percentiles
        for distance, count in data.items():
            for percentile in percentiles:
                #if percentile > prev:
                percentiles[percentile] += count
            percentiles[distance] = count
            prev = distance

        # divide by total number of distances
        percentiles = {percentile: count/len(self.distances) for percentile, count in percentiles.items()}

        return percentiles

    def plot_distances(self):
        data = Counter(self.distances)
        x = list(data.keys())
        x = [str(i) for i in x]
        y = list(data.values())
        plt.figure()
        plt.bar(x, y)
        plt.ylabel("Counts")
        plt.xlabel("Distances")
        plt.show()

    def plot_percentiles(self):
        print(Counter(self.distances))
        print(self.percentiles)
        x = list(self.percentiles.keys())
        x = ["d <= " + str(i) for i in x]
        y = list(self.percentiles.values())
        plt.figure()
        plt.bar(x, y)
        plt.xlabel("Distances")
        plt.ylabel("Percentiles")
        plt.show()


if __name__ == '__main__':
    lst1 = ["matrem", "patrem", "filium", "filia", "soror"]
    lst2 = ["matre", "patri", "fillo", "filla", "sor"]
    ld = LevenshteinDistance(lst1, lst2)
    ld.plot_distances()
    ld.plot_percentiles()
