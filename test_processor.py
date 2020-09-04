from processor import Matrix


def new_matrix():
    new_matrix = Matrix(2, 3)
    return new_matrix

def add_new_matrix():
    rows, cols = map(int, input().split())
    new_matrix = Matrix(rows, cols)
    new_matrix.input_matrix()
    return new_matrix

def scalar_multiply():
    matrix = add_new_matrix()
    const = int(input())
    print(matrix)
    matrix *= const
    print(matrix)

if __name__ == '__main__':
    new_matrix = add_new_matrix()
    for i in range(1, 4):
        sub_matrix = new_matrix.minor_submatrix(new_matrix.matrix, 1, i)
        print(sub_matrix)
    print(new_matrix)
    

    
