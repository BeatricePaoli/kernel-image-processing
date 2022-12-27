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

    points5 = np.array([(1, 3.4292989666573703), (2, 1.964930933356906), (4, 1.2800985666302342), (8, 1.466773833303402), (16, 1.4935342666382592)])
    points5[:, 1] = points5[0, 1] / points5[:, 1]

    points6 = np.array([(1, 29.415120849967934), (2, 16.59907719999319), (4, 10.596833100018557), (8, 11.87005709996447), (16, 12.598525599983986)])
    points6[:, 1] = points6[0, 1] / points6[:, 1]

    points7 = np.array([(1, 124.91958096662226), (2, 80.0472203999913), (4, 55.77724843332544), (8, 63.548848333302885), (16, 63.45572486667273)])
    points7[:, 1] = points7[0, 1] / points7[:, 1]

    points8 = np.array([(1, 189.37202040001284), (2, 121.48998019995634), (4, 85.6683819500031), (8, 97.31531704997178), (16, 96.36492774996441)])
    points8[:, 1] = points8[0, 1] / points8[:, 1]

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    # image test.jpg
    plt.title('Speedup with different kernel sizes')
    plt.plot(points1[:, 0], points1[:, 1], color='r', label='Kernel 3x3')
    plt.plot(points2[:, 0], points2[:, 1], color='g', label='Kernel 5x5')
    plt.plot(points3[:, 0], points3[:, 1], color='b', label='Kernel 7x7')
    plt.plot(points4[:, 0], points4[:, 1], color='m', label='Kernel 9x9')

    plt.legend()
    plt.savefig('speedup_by_kernel_sizes.png')

    plt.clf()

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    # kernel 7x7
    plt.title('Speedup with different image sizes')
    plt.plot(points5[:, 0], points5[:, 1], color='r', label='Image 250x250')
    plt.plot(points6[:, 0], points6[:, 1], color='g', label='Image 734x694')
    plt.plot(points7[:, 0], points7[:, 1], color='b', label='Image 1920x1080')
    plt.plot(points8[:, 0], points8[:, 1], color='m', label='Image 2048x1536')

    plt.legend()
    plt.savefig('speedup_by_img_sizes.png')
