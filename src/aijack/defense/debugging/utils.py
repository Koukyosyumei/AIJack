import numpy as np


class Hungarian:
    def __init__(self):
        self.optimal = []

    # step1
    def _step1(self, mat):
        output_mat = np.zeros_like(mat)
        for i, row in enumerate(mat):
            output_mat[i] = row - np.min(row)
        return output_mat

    # step2
    def _step2(self, mat):
        zero_coordinate = []
        for i, row in enumerate(mat):
            zero_coordinate.extend([(i, j) for j, v in enumerate(row) if v == 0])
        check_row = []
        check_column = []
        for elem in zero_coordinate:
            if not elem[0] in check_row and not elem[1] in check_column:
                check_row.append(elem[0])
                check_column.append(elem[1])
        if len(check_row) != mat.shape[0]:
            return False, zero_coordinate
        return True, zero_coordinate

    # step3
    def _step3(self, mat, zero_coordinate):
        zero_list = zero_coordinate
        zero_count = {}
        line = []
        while len(zero_list) > 0:
            for elem in zero_list:
                r = "r_" + str(elem[0])
                c = "c_" + str(elem[1])
                if r in zero_count:
                    zero_count[r] += 1
                else:
                    zero_count[r] = 1
                if c in zero_count:
                    zero_count[c] += 1
                else:
                    zero_count[c] = 1
            max_zero = max(zero_count.items(), key=lambda x: x[1])[0]
            line.append(max_zero)
            rc = max_zero.split("_")[0]
            num = max_zero.split("_")[1]
            if rc == "r":
                zero_list = [v for v in zero_list if str(v[0]) != num]
            else:
                zero_list = [v for v in zero_list if str(v[1]) != num]
            zero_count = {}
        return line

    # step4
    def _step4(self, mat, line):
        output_mat = np.zeros_like(mat)
        line_r = []
        line_c = []
        for l in line:
            rc = l.split("_")[0]
            num = int(l.split("_")[1])
            if rc == "r":
                line_r.append(num)
            else:
                line_c.append(num)
        line_cut_mat = np.delete(np.delete(mat, line_r, 0), line_c, 1)
        mini = np.min(line_cut_mat)
        cross_point = [(i, j) for i in line_r for j in line_c]
        non_line_point = [
            (i, j)
            for i in range(0, mat.shape[0])
            for j in range(0, mat.shape[0])
            if i not in line_r
            if j not in line_c
        ]
        for co in cross_point:
            mat[co] += mini
        for co in non_line_point:
            mat[co] -= mini
        return mat

    def compute(self, mat):
        mat = self._step1(mat)
        mat = self._step1(mat.T).T
        while True:
            flag, zero_coordinate = self._step2(mat)
            if flag:
                break
            line = self._step3(mat, zero_coordinate)
            mat = self._step4(mat, line)
        r = []
        c = []
        for v in zero_coordinate:
            if v[0] not in r and v[1] not in c:
                self.optimal.append(v)
                r.append(v[0])
                c.append(v[1])
        return self.optimal
