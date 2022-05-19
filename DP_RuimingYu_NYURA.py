import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pip import List
from sklearn.linear_model import LinearRegression
from copy import copy, deepcopy
from numpy import linalg as LA
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from scipy.stats import norm
import collections
import time
import statistics
import math
import scipy.integrate as integrate
from scipy.optimize import bisect

class Dataset:
    def __init__(self, sigma=1, cu=5, co=1, length=500) -> None:
        # from input
        self.sigma = sigma
        self.cu = cu
        self.co = co
        self.length = length
        self.unit_cost = co

        # from input with manipulation
        self.sale_price = cu + co
        self.phi = norm.ppf(cu / (cu + co))
        self.ratio = cu / (cu + co)
    
    
    def process(self) -> None:
        """
        this function is used to go over all the steps in the model,
        but in real processes, we need to repeat some parts yet others require only one run.
        """
        time_start = time.time()
        self.__initiate_data_x_y()
        print("part 1 finished")
        print("created self.x, self.y, self.scaler \n")

        self.cal_max_beta()
        print("part 2 finished")
        print("calculated self.max_beta \n")

        self.regression_experiment()
        print("part 3 finished")
        print("created self.beta, self.intercept, self.se \n")

        self.do_exp()
        print("part 4&5 is finished")
        print("experiment finished \n")

        self.theory_bound()
        print("part 6 is finished")
        print("intergral finished \n")

        time_end = time.time()
        print("Time elapsed:", time_end - time_start, "\n")
        
        return


# Part 1: generate the `x` and `y`

    def __initiate_data_x_y(self) -> None:
        """
        generate the x and y
        """
        self.__generate_x_initial()
        self.scaler = preprocessing.MinMaxScaler().fit(self.x_initial)
        self.x = collections.deque(self.scaler.transform(self.x_initial))
        self.y = collections.deque([self.__demand(x) for x in self.x])
        return


    def __generate_x_initial(self, center=[0, 0], cov_matrix=[[1, 0], [0, 1]], length=500) -> None:
        self.x_initial = multivariate_normal.rvs(center, cov_matrix, size = length)
        return


    def __demand(self, x_list_list: List[List[float]]) -> float:
        return 20 * x_list_list[0] - 10 * x_list_list[1] + norm.rvs(scale=self.sigma, size=1)[0] + 10


    def plot_initial(self) -> None:
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.x[:, 0], self.x[:, 1], self.y, 'gray')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Demand')
        plt.show()
        return


# Part 2: calculate the `max_beta`

    def cal_max_beta(self) -> None:
        outlier = self.__find_outlier()
        self.max_beta = self.__find_max_beta(outlier)
        return


    def __find_outlier(self) -> int:
        length_y = len(self.y)
        min_beta = float('inf')
        outlier = -1
        
        # use deque to pop one group of x and y each time, and fit the model.
        # append the popped element to the end of the deque
        # so that we perform a leave-one-out method
        for i in range(length_y):
            x_leave = self.x.popleft()
            y_leave = self.y.popleft()

            cur_model = LinearRegression().fit(self.x, self.y)
            b1 = cur_model.intercept_
            b2 = cur_model.coef_
            b_size = math.sqrt(b1 ** 2 + sum(b2 ** 2))

            if min_beta > b_size:
                min_beta = b_size
                outlier = i

            self.x.append(x_leave)
            self.y.append(y_leave)

        return outlier


    def __find_max_beta(self, outlier: int) -> float:
        length_y = len(self.y)
        max_beta = -1
        outlier_x = self.x[outlier]
        outlier_y = self.y[outlier]
        self.x.append(outlier_x)
        self.y.append(outlier_y)

        for i in range(length_y):
            x_leave = self.x.popleft()
            y_leave = self.y.popleft()

            model = LinearRegression().fit(self.x, self.y)
            b1 = model.intercept_
            b2 = model.coef_
            b_size = np.sqrt(b1 ** 2 + sum(b2 ** 2))

            if max_beta <= b_size:
                max_beta = b_size
            
            self.x.append(x_leave)
            self.y.append(y_leave)
        
        # as we have appended the outlier in the beginning, we pop it now.
        # the popleft-append manipulation is done for n times, so we are
        # sure that the outlier section is at the left-most of the deque.
        self.x.popleft()
        self.y.popleft()
        return max_beta
            

# Part 3: setup for the regression, compute the coef and intercept

    def regression_experiment(self) -> None:
        model = LinearRegression().fit(self.x, self.y)
        y_pred = model.predict(self.x)
        self.beta = model.coef_
        self.intercept = model.intercept_
        self.se = np.sqrt(sum((y_pred - self.y) ** 2) / (len(y_pred) - 1))
        return

# Part 4: setup for part 5, create several functions for different situations

    def cal_optimal_quantity(self, x_list: List[float]) -> float:
        return sum(self.beta * x_list) + self.intercept + self.se * self.phi
    

    def cal_rand_q(self, x_list: List[float], ep1=0.5, ep2=0.5) -> float:
        sen1 = 4 * np.sqrt(2) * (self.max_beta + max(np.abs(self.y))) / (len(self.y)) / ep1
        sen2 = (self.max_beta + max(np.abs(self.y))) / np.sqrt(len(self.y) - 1) / ep2

        return self.cal_optimal_quantity(x_list) + np.random.laplace(loc=0.0, scale=sen1) + np.abs(self.phi) * np.random.laplace(
            loc=0.0, scale=sen2)


    def cal_profit(self, q_input: float, d_input: float) -> float:
        val = min(q_input, d_input) * self.sale_price - q_input * self.unit_cost
        return val


    def cal_cost(self, q_input: float, d_input: float) -> float:
        if q_input >= d_input:
            val = self.co * (q_input - d_input)
        else:
            val = self.cu * (d_input - q_input)
        return val


    def __generate_current_data(self) -> List[float]:
        try_x = multivariate_normal.rvs([0, 0], [[1, 0], [0, 1]], size=1)
        return self.scaler.transform([try_x])[0]


# Part 5: three types of experiment

    def do_exp(self, exp_size=1, full_info_size=100, dp_info_size=100, no_info_size=100) -> None:
        self.__clear_do_list()
        
        for _ in range(exp_size):
            cur_x = self.__generate_current_data()

            self.do_full_info(full_info_size, cur_x)
            self.do_dp_info(dp_info_size, cur_x)
            self.do_no_info(no_info_size, cur_x)

        self.profit_dp_info_list_agg = list(
            sum(np.array(self.profit_dp_info_list_agg)) / len(self.profit_dp_info_list_agg))
        self.cost_dp_info_list_agg = list(
            sum(np.array(self.cost_dp_info_list_agg)) / len(self.cost_dp_info_list_agg))
        return


    def do_full_info(self, size=100, cur_x = [0.0]) -> None:
        """
        experiment with full information
        """
        cost_full_info = 0
        profit_full_info = 0

        for _ in range(size):
            # `self.__demand()` involves randomness so it has to be included in loop
            d = self.__demand(cur_x)
            q = self.cal_optimal_quantity(cur_x)

            profit_full_info += self.cal_profit(q, d)
            cost_full_info += self.cal_cost(q, d)

        avg_cost_full_info = cost_full_info / size
        avg_profit_full_info = profit_full_info / size

        self.cost_full_info_list.append(avg_cost_full_info)
        self.profit_full_info_list.append(avg_profit_full_info)

        return


    def do_dp_info(self, size=100, cur_x = [0.0]) -> None:
        """
        experiment with differential privacy
        """
        cost_dp_info_list = []
        profit_dp_info_list = []

        for ep_iterator in range(1, 30):
            ep = ep_iterator / 5
            cost_dp_info = 0
            profit_dp_info = 0

            for _ in range(size):
                d = self.__demand(cur_x)
                q = self.cal_rand_q(cur_x, ep * 0.1, ep * 0.9)

                profit_dp_info += self.cal_profit(q, d)
                cost_dp_info += self.cal_cost(q, d)

            avg_cost_dp_info = cost_dp_info / size
            avg_profit_dp_info = profit_dp_info / size
            cost_dp_info_list.append(avg_cost_dp_info)
            profit_dp_info_list.append(avg_profit_dp_info)
        
        self.cost_dp_info_list_agg.append(cost_dp_info_list[:])
        self.profit_dp_info_list_agg.append(profit_dp_info_list[:])

        return


    def do_no_info(self, size=100, cur_x = [0.0]) -> None:
        """
        experiment for no-information
        """
        # `self.y` is collection.deque, which cannot be sorted
        copy_sort_y = list(self.y)
        copy_sort_y.sort()
        cost_no_info = 0
        profit_no_info = 0
        
        target_index = int(self.cu / (self.cu + self.co) * len(copy_sort_y)) - 1
        q = copy_sort_y[target_index]

        for _ in range(size):
            d = self.__demand(cur_x)

            profit_no_info += self.cal_profit(q, d)
            cost_no_info += self.cal_cost(q, d)

        avg_cost_no_info = cost_no_info / size
        avg_profit_no_info = profit_no_info / size

        self.cost_no_info_list.append(avg_cost_no_info)
        self.profit_no_info_list.append(avg_profit_no_info)
        return

    
    def __clear_do_list(self) -> None:
        self.profit_no_info_list: List[float] = []
        self.cost_no_info_list: List[float] = []
        self.profit_dp_info_list_agg: List[List[float]] = []
        self.cost_dp_info_list_agg: List[List[float]] = []
        self.profit_full_info_list: List[float] = []
        self.cost_full_info_list: List[float] = []
        return
    
    # Part 6: intergral

    def plot_profit(self) -> None:
        """
        plot profit
        """
        plt.plot(np.arange(1, 30) / 5,
                 [sum(self.profit_full_info_list) / len(self.profit_full_info_list) for _ in range(29)],
                 label='Optimal profit', c='r')

        plt.plot(np.arange(1, 30) / 5, self.profit_dp_info_list_agg, label='DP profit', c='k')

        plt.plot(np.arange(1, 30) / 5,
                 [sum(self.profit_no_info_list) / len(self.profit_no_info_list) for _ in range(29)],
                 label='No information profit', c='0.3')

        plt.legend(loc='best')
        plt.xlabel('privacy loss ϵ')
        plt.ylabel('Average Profit with Respect to optimal profit')
        # plt.title('Profit Comparison between optimal ordering, differential private ordering and no-information ordering')
        plt.show()
        return

    def theory_bound(self) -> List[float]:

        #long run time ~180 sec
        bound_list = []
        for ep_iterator in range(1, 30):
            ep = ep_iterator / 5
            self.s1 = self.cal_sen1(ep)
            self.s2 = self.cal_sen2(ep)

            integral_ = integrate.quad(self.prob_loss, 0, np.infty)
            integral = integral_[0]/(self.s1 ** 2 - self.s2 ** 2)/2
            bound_list.append(integral)
        
        print(bound_list)
        return bound_list

    def cal_sen1(self, ep) -> float:
        return 4 * np.sqrt(2) * (self.max_beta + max(self.y)) / (len(self.y)) / (ep * 0.1)

    def cal_sen2(self, ep) -> float:
        return np.abs(self.phi) * (self.max_beta + max(np.abs(self.y))) / np.sqrt(len(self.y) - 1) / (ep * 0.9)

    def prob_loss(self, alpha: float) -> float:
        q1 = bisect(lambda x: x - alpha / self.sale_price / (norm.cdf(x + self.phi) - self.ratio),
                    -2 * 1e6, -1e-7, xtol=1e-6)
        q2 = bisect(lambda x: x - alpha / self.sale_price / (norm.cdf(x + self.phi) - self.ratio),
                    1e-7, 2 * 1e6, xtol=1e-6)
        prob = self.s1 ** 2 * (np.exp(q1 / self.s1) + np.exp(-q2 / self.s1)) - self.s2 ** 2 * (np.exp(q1 / self.s2) + np.exp(-q2 / self.s2))
        return prob