from scripts.folklores import load_folklore
from gtda.diagrams import PairwiseDistance
from gtda.plotting import plot_heatmap
from gtda.plotting import plot_point_cloud
import matplotlib.pyplot as plt


def get_persistent_homology(df, cols=None):
    if cols is None:
        cols = ["SBERT", "LF"]

    emb2diagram = {}
    for emb in cols:
        cos_matrix = util.pytorch_cos_sim(df[emb], df[emb])
        # requires input as 3d array
        dis_matrix = 1 - cos_matrix[np.newaxis, :, :]

        homology_dimensions = [0, 1, 2]

        persistence = hl.VietorisRipsPersistence(
            metric="precomputed", homology_dimensions=homology_dimensions
        )

        # persistence_diagram[i] = topology of dis_matrix[i]
        emb2diagram[emb] = persistence.fit_transform_plot(dis_matrix)

    return emb2diagram

def compute_distance(emb2diagram, metric: str = "bottleneck"):
    dis = PairwiseDistance(metric=metric)
    dis.fit(emb2diagram["SBert"])
    dis.transform(emb2diagram["LF"])
    return dis

def plot_lifetime(persistence_diagram):
    """lifetime for H0 - x and H1 - y"""
    diagram = persistence_diagram[0]
    lifetimes = []
    for dim in [0, 1]:
        data = diagram[diagram[:,-1] == dim]
        lifetime = data[:,1] - data[:, 0]
        lifetimes.append(lifetime)
    plt.scatter(x=lifetimes[0], y=[0] * len(lifetimes[0]))
    plt.scatter(x=[0] * len(lifetimes[1]), y=lifetimes[1])
    plt.xlabel("$H_0$")
    plt.ylabel("$H_1$")


if __name__ == '__main__':
    folklore_df = load_folklore(True)
    emb2diagram = get_persistent_homology(folklore_df)
    persistence_diagram = emb2diagram["LF"]
    plot_lifetime(persistence_diagram)
    plot_point_cloud(persistence_diagram[0])
    plot_point_cloud(persistence_diagram[0, 1:], dimension=2)


    # tri heatmap of cos matrix
    emb = "LF"
    cos_matrix = util.pytorch_cos_sim(folklore_df[emb], folklore_df[emb])
    f, ax = plt.subplots(figsize=(18, 18))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.zeros_like(cos_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        cos_matrix,
        mask=mask, square=True,
        cmap=cmap, center=0.1
    )
    # OR
    cos_matrix_upper = cos_matrix.copy()[mask] = None
    plot_heatmap(cos_matrix, plotly_params=dict(yaxis_autorange='reversed'))