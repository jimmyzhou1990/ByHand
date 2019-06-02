import numpy as np

def kernel_linear(xi, xj):
    return np.dot(xi.transpose(), xj)

def kernel_quadratic(xi, xj):
    xi_t = xi.copy()
    xi_t[0] = np.power(xi_t[0], 2)
    xj_t = xj.copy()
    xj_t[0] = np.power(xj_t[0], 2)
    return np.dot(xi_t.transpose(), xj_t)

class SVM(object):
    def __init__(self, input_x, input_y, C=10000, kernel='linear'):
        self.X = input_x
        self.Y = input_y
        self.N, self.dim = input_x.shape
        self.b = 0
        self.C = C
        self.alpha = np.zeros(self.N)
        self.supportIndex = set()
        self.E = np.zeros(self.N)
        self.tol = 0.0001
        self.eps = 0.01
        self.w = np.zeros(input_x.shape[1])
        self.step_count = 0
        if kernel == 'linear':
            self.kernel = kernel_linear
        elif kernel == 'quadratic':
            self.kernel = kernel_quadratic

    def update_E(self):
        for i in range(self.N):
            self.E[i] = self.Gx(i) - self.Y[i]

    def find_second_alpha_index(self, i2):
        abs_ = -1
        i1 = -1
        for index in self.supportIndex:
            tmp_abs = np.abs(self.E[i2] - self.E[index])
            if tmp_abs > abs_:
                i1 = index
                abs_ = tmp_abs
        return i1

    def objective_at_alpha(self, alpha1, alpha2, i1, i2):
        w = 0
        w += 0.5*self.kernel(self.X[i2], self.X[i2])*alpha2*alpha2
        w += self.Y[i1]*self.Y[i2]*self.kernel(self.X[i1], self.X[i2])*alpha1*alpha2
        w -= alpha2

        s = 0
        for i in range(self.N):
            if i != i1 and i != i2:
                s += self.Y[i]*self.alpha[i]*self.kernel(self.X[i2], self.X[i])
        s *= self.Y[i2]*alpha2
        w += s

        return w


    def takeStep(self, i1, i2):
        if i1 == i2:
            return  False

        alpha1 = self.alpha[i1]
        y1 = self.Y[i1]
        E1 = self.E[i1]
        alpha2 = self.alpha[i2]
        y2 = self.Y[i2]
        E2 = self.E[i2]

        s = y1 * y2
        if s > 0:
            L = np.max([0, alpha1+alpha2-self.C])
            H = np.min([self.C, alpha1+alpha2])
        else:
            L = np.max([0, alpha2-alpha1])
            H = np.min([self.C, self.C+alpha2-alpha1])
        if L == H:
            return False

        K11 = self.kernel(self.X[i1], self.X[i1])
        K12 = self.kernel(self.X[i1], self.X[i2])
        K22 = self.kernel(self.X[i2], self.X[i2])
        eta = K11 + K22 - 2 * K12
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            Lobj = self.objective_at_alpha(alpha1, L, i1, i2)
            Hobj = self.objective_at_alpha(alpha1, H, i1, i2)
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj+self.eps:
                a2 = H
            else:
                a2 = alpha2

        if np.abs(a2-alpha2) < self.eps * (a2+alpha2+self.eps):
            return False

        a1 = alpha1 + s * (alpha2 - a2)
        b1_new = -E1 - y1*K11*(a1 - alpha1) - y2*K12*(a2 - alpha2) + self.b
        b2_new = -E2 - y1*K12*(a1 - alpha1) - y2*K22*(a2 - alpha2) + self.b
        if (0 < a1 < self.C) and (0 < a2 < self.C):
            self.b = b1_new
        else:
            self.b = (b1_new + b2_new)/2
        self.alpha[i2] = a2
        self.alpha[i1] = a1
        self.update_E()
        self.update_support(i1, i2)
        self.step_count += 1
        return True

    def update_support(self, i1, i2):
        if 0 < self.alpha[i1] < self.C:
            self.supportIndex.add(i1)
        elif i1 in self.supportIndex:
            self.supportIndex.remove(i1)

        if 0 < self.alpha[i2] < self.C:
            self.supportIndex.add(i2)
        elif i2 in self.supportIndex:
            self.supportIndex.remove(i2)

    def examine_example(self, i2):
        y2 = self.Y[i2]
        alph2 = self.alpha[i2]
        E2 = self.E[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):
            if len(self.supportIndex) > 1:
                i1 = self.find_second_alpha_index(i2)
                if self.takeStep(i1, i2):
                    return 1
            for index in self.supportIndex:
                i1 = index
                if self.takeStep(i1, i2):
                    return 1

            for index in range(self.N):
                i1 = index
                if self.takeStep(i1, i2):
                    return 1

        return 0


    def train(self):
        numChanged = 0
        examineAll = True
        self.update_E()
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(self.N):
                    numChanged += self.examine_example(i)
            else:
                for i in range(self.N):
                    if 0 < self.alpha[i] < self.C:
                        numChanged += self.examine_example(i)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

        self.w = self.weight()

    def Gx(self, i):
        g = 0
        for j in range(self.N):
            g += self.alpha[j]*self.Y[j]*self.kernel(self.X[j], self.X[i])
        g += self.b
        return g

    def inference(self, x):
        pass

    def weight(self):
        if self.kernel is kernel_linear:
            w = np.zeros(self.X.shape[1])
            for j in range(self.N):
                w += self.alpha[j]*self.Y[j]*self.X[j]
        elif self.kernel is kernel_quadratic:
            w = np.zeros(self.X.shape[1])
            for j in range(self.N):
                w[0] += self.alpha[j]*self.Y[j]*np.power(self.X[j][0], 2)
                w[1] += self.alpha[j]*self.Y[j]*self.X[j][1]
        return w



