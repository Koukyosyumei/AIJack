import numpy as np


class Hungarian:
    """
    Hungarian algorithm implementation for solving the assignment problem.
    """

    def __init__(self):
        """
        Initializes the Hungarian algorithm.
        """
        self.optimal = []

    # step1
    def _step1(self, mat):
        """
        Performs step 1 of the Hungarian algorithm.
        Subtracts the minimum value of each row from each element of the row,
        and then subtract the minimum value of each column from each element of the column.

        Args:
            mat (numpy.ndarray): The input matrix.

        Returns:
            numpy.ndarray: The output matrix after step 1.
        """
        output_mat = np.zeros_like(mat)
        for i, row in enumerate(mat):
            output_mat[i] = row - np.min(row)
        return output_mat

    # step2
    def _step2(self, mat):
        """
        Performs step 2 of the Hungarian algorithm.
        Determines whether it is possible to select one zero from each row and column.
        If it is possible, the coordinates of the selected zeros are returned as assignment candidates.
        If it is not possible, proceed to step 3.

        Args:
            mat (numpy.ndarray): The input matrix.

        Returns:
            tuple: A tuple containing a boolean flag indicating whether step 2 is completed,
                and a list of zero coordinates.
        """
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
        """
        Performs step 3 of the Hungarian algorithm.
        Covers all zeros with the minimum number of horizontal or vertical lines.

        Args:
            mat (numpy.ndarray): The input matrix.
            zero_coordinate (list): A list of zero coordinates.

        Returns:
            list: A list of lines.
        """
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
        """
        Performs step 4 of the Hungarian algorithm.
        Subtracts the minimum value from the elements not covered by the lines,
        and add the value to the elements where the lines intersect.
        Then, go back to step 2.

        Args:
            mat (numpy.ndarray): The input matrix.
            line (list): A list of lines.

        Returns:
            numpy.ndarray: The updated matrix after step 4.
        """
        # output_mat = np.zeros_like(mat)
        line_r = []
        line_c = []
        for el in line:
            rc = el.split("_")[0]
            num = int(el.split("_")[1])
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
        """
        Computes the optimal assignment using the Hungarian algorithm.

        Args:
            mat (numpy.ndarray): The input matrix.

        Returns:
            list: The list of optimal assignments.
        """
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
