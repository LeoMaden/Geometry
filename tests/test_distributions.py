import matplotlib.pyplot as plt

from geometry.distributions import cluster_left, cluster_right, double_sided


def main():
    # Left point clustering with fixed expansion ratio
    plt.figure()
    plt.plot(cluster_left(0, 1, 20, r=1.1), "ro", label=r"$r=1.1$")
    plt.plot(cluster_left(0, 1, 20, r=1.2), "bo", label=r"$r=1.2$")
    plt.legend()
    plt.grid()

    # Left point clustering with fixed initial step size
    plt.figure()
    plt.plot(cluster_left(0, 1, 20, delta=0.01), "ro", label=r"$\delta=0.01$")
    plt.plot(cluster_left(0, 1, 20, delta=0.001), "bo", label=r"$\delta=0.001$")
    plt.legend()
    plt.grid()

    # Right point clustering
    plt.figure()
    plt.plot(cluster_right(0, 1, 20, delta=0.01), "ro", label=r"$\delta=0.01$")
    plt.plot(cluster_right(0, 1, 20, r=1.4), "bo", label=r"$r=1.4$")
    plt.legend()
    plt.grid()

    # Double sided clustering
    plt.figure()
    plt.plot(
        double_sided(0, 1, 20, 0.2, delta=0.01),
        "ro",
        label=r"$\alpha=0.20, \delta=0.01$",
    )
    plt.plot(
        double_sided(0, 1, 20, 0.35, delta=0.01),
        "bo",
        label=r"$\alpha=0.35, \delta=0.01$",
    )
    plt.plot(
        double_sided(0, 1, 20, 0.2, r=1.1),
        "go",
        label=r"$\alpha=0.20, r=1.1$",
    )
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
