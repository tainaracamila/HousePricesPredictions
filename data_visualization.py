import matplotlib.pyplot as plt

class Graphs(object):

    def scatter(self, x, y, x_label, y_label, title):
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def double_scatter(self, pred_x_train, pred_x_test, y_train, y_test):
        plt.scatter(pred_x_train, pred_x_train - y_train, c='b', s=40, alpha=0.5)
        plt.scatter(pred_x_test, pred_x_test - y_test, c='g', s=40, alpha=0.5)
        plt.hlines(y=0, xmin=0, xmax=50)
        plt.ylabel('Residue')
        plt.title('Residual plot - Blue = Train and Green = Test')
        plt.show()



