from copy import copy, deepcopy


class MatrixDimensionError(Exception):
    pass


class MatrixInverseNoExistException(Exception):
    pass


class Matrix:
    """class for matrix"""
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.matrix = self._init_zero_matrix()

    def __str__(self):
        str_matrix = ''
        for row in self.matrix:
            if str_matrix:
                str_matrix = f'{str_matrix} \n{" ".join(map(str, row))}'
            else:
                str_matrix = ' '.join(map(str, row))
        return str_matrix

    def __copy__(self):
        new_one = type(self)(self.rows, self.cols)
        new_one.__dict__.update(self.__dict__)
        return new_one

    def __add__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            new_matrix = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix.matrix[i][j] = self.matrix[i][j] +\
                                              other.matrix[i][j]
            return new_matrix
        else:
            raise MatrixDimensionError

    def __mul__(self, other):
        if self.cols == other.rows:
            new_matrix = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    row = self.matrix[i]
                    col = [other.matrix[k][j] for k in range(other.rows)]
                    new_matrix.matrix[i][j] = \
                        sum(map(lambda x: x[0] * x[1], zip(row, col)))
            return new_matrix
        else:
            raise MatrixDimensionError

    def __imul__(self, const: int):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = self.matrix[i][j] * const
        return self

    @staticmethod
    def minor_matrix(matrix, row_index, col_index):
        """get a submatrix for minor for row index and column index"""
        minor_matrix = deepcopy(matrix)
        minor_matrix.pop(row_index - 1)
        for row in minor_matrix:
            row.pop(col_index - 1)
        return minor_matrix

    @classmethod
    def get_determinant(cls, matrix):
        """evaluate determinant recursively with Laplace expansion"""
        size = len(matrix)
        if size == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        elif size == 1:
            return matrix[0][0]
        else:
            det = 0
            for i in range(1, size + 1):
                det += matrix[0][i - 1] * (-1) ** (1 + i)\
                      * cls.get_determinant(cls.minor_matrix(matrix, 1, i))
            return det

    def _init_zero_matrix(self):
        """init zero matrix with rows, cols"""
        matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        return matrix

    def input_matrix(self):
        self.matrix = \
            [[float(x) for x in input('> ').split()] for _ in range(self.rows)]

    def determinant(self):
        if self.rows != self.cols:
            raise MatrixDimensionError
        else:
            return self.__class__.get_determinant(self.matrix)

    def transposition(self):
        new_matrix = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix.matrix[j][i] = self.matrix[i][j]
        return new_matrix

    def transposition_side(self):
        new_matrix = self.transposition()
        new_matrix.matrix.reverse()
        for row in new_matrix.matrix:
            row.reverse()
        return new_matrix

    def transposition_vertical(self):
        new_matrix = copy(self)
        for row in new_matrix.matrix:
            row.reverse()
        return new_matrix

    def transposition_horizontal(self):
        new_matrix = copy(self)
        new_matrix.matrix.reverse()
        return new_matrix

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise MatrixInverseNoExistException
        else:
            c_matrix = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    minor = Matrix.minor_matrix(self.matrix, i + 1, j + 1)
                    c_matrix.matrix[i][j] = (-1) ** (i + j + 2)\
                                            * Matrix.get_determinant(minor)
            c_matrix = c_matrix.transposition()
            c_matrix *= 1 / det
            return c_matrix


def add_new_matrix(number_matrix: str):
    message = f'Enter size of {number_matrix}matrix: > '
    rows, cols = map(int, input(message).split())
    message = f'Enter {number_matrix}matrix:'
    new_matrix = Matrix(rows, cols)
    print(message)
    new_matrix.input_matrix()
    return new_matrix


def add_two_matrix():
    matrix_a = add_new_matrix('first ')
    matrix_b = add_new_matrix('second ')
    try:
        matrix_c = matrix_a + matrix_b
        print('The result is:')
        print(matrix_c)
    except MatrixDimensionError:
        print('The operation cannot be performed.')
    return True


def scalar_multiply():
    matrix = add_new_matrix('')
    const = float(input('Enter constant: > '))
    matrix *= const
    print('The result is:')
    print(matrix)
    return True


def mul_two_matrix():
    matrix_a = add_new_matrix('first ')
    matrix_b = add_new_matrix('second ')
    try:
        matrix_c = matrix_a * matrix_b
        print('The result is:')
        print(matrix_c)
    except IndexError as exp:
        print('The operation cannot be performed.')
    return True


def trans_main_diagonal(matrix):
    matrix_t = matrix.transposition()
    print('The result is:')
    print(matrix_t)


def trans_side_diagonal(matrix):
    matrix_t = matrix.transposition_side()
    print('The result is:')
    print(matrix_t)


def trans_vertical_line(matrix):
    matrix_t = matrix.transposition_vertical()
    print('The result is:')
    print(matrix_t)


def trans_horizontal_line(matrix):
    matrix_t = matrix.transposition_horizontal()
    print('The result is:')
    print(matrix_t)


def calc_determinant():
    matrix = add_new_matrix('')
    try:
        det = matrix.determinant()
        print('The result is:')
        print(det)
    except MatrixDimensionError:
        print('The operation cannot be performed.')
    return True


def inverse_matrix():
    matrix = add_new_matrix('')
    try:
        inv_matrix = matrix.inverse()
        print('The result is:')
        print(inv_matrix)
    except MatrixDimensionError:
        print('The operation cannot be performed.')
    except MatrixInverseNoExistException:
        print("This matrix doesn't have an inverse.")
    return True


def bye():
    """return False'"""
    return False


def user_menu() -> int:
    """print user menu, wait command"""
    print("1. Add matrices",
          "2. Multiply matrix by a constant",
          "3. Multiply matrices",
          "4. Transpose matrix",
          "5. Calculate a determinant",
          "6. Inverse matrix",
          "0. Exit", sep='\n')
    command = ' '
    while command not in '0123456':
        command = input('Your choice: > ')
    return int(command)


def transposition_menu():
    """print transposition menu, wait choice"""
    print("",
          "1. Main diagonal",
          "2. Side diagonal",
          "3. Vertical line",
          "4. Horizontal line", sep='\n')
    choice = ' '
    while choice not in '1234':
        choice = input('Your choice: > ')
    return int(choice)


def trans_matrix():
    trans_menu = {
        1: trans_main_diagonal,
        2: trans_side_diagonal,
        3: trans_vertical_line,
        4: trans_horizontal_line
    }
    trans_command = transposition_menu()
    select_trans = trans_menu[trans_command]
    matrix = add_new_matrix('')
    select_trans(matrix)
    return True


if __name__ == '__main__':
    menu = {
        0: bye,
        1: add_two_matrix,
        2: scalar_multiply,
        3: mul_two_matrix,
        4: trans_matrix,
        5: calc_determinant,
        6: inverse_matrix
    }
    working = True
    while working:
        user_command = user_menu()
        select_command = menu[user_command]
        working = select_command()
