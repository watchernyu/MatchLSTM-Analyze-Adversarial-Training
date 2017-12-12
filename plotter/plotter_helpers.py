colors = ["#0066ff","#00ffff","#00ff00","#ffff99","#ff9966","#ff3300","#660033"] #original, end, front, any1, any2, any3, any4
plot_folder_name = "../plotlogs/firstround_results"

def get_colomn(lines,index):
    n = len(lines)
    colomn = []
    for i in range(n):
        colomn.append(lines[i][index])
    return colomn

def get_array(r,c):
    newarray = [[0 for _ in range(c)] for _ in range(r)]
    return newarray