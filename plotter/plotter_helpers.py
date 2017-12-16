from matplotlib import pyplot as plt

colors = ["#0066ff","#00ffff","#00ff00","#ffff99","#ff9966","#ff3300","#660033"] #original, end, front, any1, any2, any3, any4

def get_colomn(lines,index):
    n = len(lines)
    colomn = []
    for i in range(n):
        colomn.append(lines[i][index])
    return colomn

def get_array(r,c):
    newarray = [[0 for _ in range(c)] for _ in range(r)]
    return newarray


def draw_table(data,modelnames):
    figure, axs = plt.subplots()

    # Hide axes
    axs.xaxis.set_visible(False)
    axs.yaxis.set_visible(False)

    # data = [[66386, 174296, 75131, 577908],
    #         [58230, 381139, 78045, 99308],
    #         [89135, 80552, 152558, 497981],
    #         [78415, 81858, 150656, 193263],
    #         [139361, 331509, 343164, 781380]]

    columns = ('Original', 'Any1', 'AddOneSent', 'AddBestSent')
    rows = modelnames

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=data,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='top')
    plt.subplots_adjust(left=0.2, top=0.7)
    plt.show()
