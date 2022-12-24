import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    points1 = np.array([(1, 6.774005200015381), (2, 3.817944250011351), (4, 2.425146549998317), (8, 2.7164076500339434), (16, 2.7332648999872617)])
    points1[:, 1] = points1[0, 1] / points1[:, 1]

    points2 = np.array([(1, 15.947121400036849), (2, 8.707318849978037), (4, 5.577878200041596), (8, 6.185469049960375), (16, 6.313950500043575)])
    points2[:, 1] = points2[0, 1] / points2[:, 1]

    points3 = np.array([(1, 29.415120849967934), (2, 16.59907719999319), (4, 10.596833100018557), (8, 11.87005709996447), (16, 12.598525599983986)])
    points3[:, 1] = points3[0, 1] / points3[:, 1]

    points4 = np.array([(1, 47.220073199947365), (2, 26.36768500006292), (4, 16.22026009997353), (8, 18.629134300048463), (16, 18.992713700048625)])
    points4[:, 1] = points4[0, 1] / points4[:, 1]

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    plt.title('Speedup with different kernel sizes')
    plt.plot(points1[:, 0], points1[:, 1], color='r', label='Kernel 3x3')
    plt.plot(points2[:, 0], points2[:, 1], color='g', label='Kernel 5x5')
    plt.plot(points3[:, 0], points3[:, 1], color='b', label='Kernel 7x7')
    plt.plot(points4[:, 0], points4[:, 1], color='m', label='Kernel 9x9')

    plt.legend()
    plt.savefig('speedup.png')
