import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import folium
from fuzzywuzzy import fuzz
import warnings
from sklearn.exceptions import ConvergenceWarning
import random
import webbrowser
from matplotlib.animation import FuncAnimation

def fuzzy_group(locations, threshold=80):
    grouped_locations = []
    for loc in locations:
        loc = str(loc)  # Ensure the location is a string
        found = False
        for group in grouped_locations:
            similarity = fuzz.ratio(loc, group[0])
            print(f"Comparing '{loc}' with '{group[0]}': Similarity = {similarity}")
            if similarity >= threshold:
                group.append(loc)
                found = True
                break
        if not found:
            grouped_locations.append([loc])
    location_mapping = {loc: group[0] for group in grouped_locations for loc in group}
    return location_mapping


def gap_statistic(x, kmax=20, b=20):
    gaps = []
    wks = []
    wkbs = []
    for k in range(1, kmax + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=27, max_iter=500, tol=1e-4).fit(x)
            inertia = kmeans.inertia_
            if inertia > 0:
                wk = np.log(inertia)
            else:
                wk = 0  # Handle zero or negative inertia
            wks.append(wk)

            ref_disps = np.zeros(b)
            for i in range(b):
                random_reference = np.random.uniform(np.min(x), np.max(x), x.shape)
                ref_kmeans = KMeans(n_clusters=k, random_state=42).fit(random_reference)
                ref_inertia = ref_kmeans.inertia_
                if ref_inertia > 0:
                    ref_disps[i] = np.log(ref_inertia)
                else:
                    ref_disps[i] = 0  # Handle zero or negative inertia

            wkb = np.mean(ref_disps)
            wkbs.append(wkb)
            gaps.append(wkb - wk)

    gaps = np.array(gaps)
    if len(gaps) > 1:
        optimal_k = np.argmax(gaps[:-1] - gaps[1:] + np.log(kmax)) + 1
    else:
        optimal_k = 1  # Default to 1 if gaps array is too small
    print(f"Gaps: {gaps}")
    print(f"Optimal k (gap statistic): {optimal_k}")
    return gaps, optimal_k


def find_optimal_clusters(X, max_k=50, plot=False, random_state=42):
    n_samples = X.shape[0]
    if max_k >= n_samples:
        max_k = n_samples - 1

    gaps, gap_optimal_k = gap_statistic(X, max_k)
    print(f"Gaps: {gaps}, Optimal k (gap statistic): {gap_optimal_k}")

    kmeans = KMeans(n_clusters=gap_optimal_k, random_state=random_state, n_init=27, max_iter=500, tol=1e-4)
    kmeans.fit(X)
    best_labels = kmeans.labels_

    if plot:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 1, 1)
        plt.plot(range(1, len(gaps) + 1), gaps, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic Method')

        plt.tight_layout()
        plt.show()

    return gap_optimal_k, best_labels


def perform_kmeans(data, n_clusters, algorithm, init, max_iter=300, noise_level=0.0):
    # Add noise to the data
    noisy_data = data[["AREA CODE"]].values + noise_level * np.random.randn(*data[["AREA CODE"]].shape)

    kmeans = KMeans(n_clusters=n_clusters, algorithm=algorithm, init=init, max_iter=max_iter, random_state=42)
    labels = kmeans.fit_predict(noisy_data)
    data["cluster"] = labels

    return labels


def visualize_kmeans(data, n_clusters, algorithm, init, max_iter=300, noise_level=0.0):
    noisy_data = data[["LAT", "LON"]].values + noise_level * np.random.randn(*data[["LAT", "LON"]].shape)
    maxval = noisy_data.max(axis=0).max()
    normalized_data = noisy_data / maxval

    fig, ax = plt.subplots()

    class KMeansVisualizer:
        def __init__(self, n_clusters, data, algorithm, init, max_iter):
            self.kmeans = KMeans(n_clusters=n_clusters, algorithm=algorithm, init=init, max_iter=max_iter, random_state=42)
            self.data = data
            self.n_clusters = n_clusters
            self.labels = np.zeros(len(data))
            self.centroids = np.random.rand(n_clusters, 2)
            self.max_iter = max_iter

        def _get_closest_class_id(self, X):
            d = ((X - self.centroids) ** 2).sum(axis=1)
            return np.argmin(d)

        def fit(self, frame):
            ax.clear()
            if self.data.shape[1] < 2:
                raise ValueError("Data must have at least two columns for visualization.")

            ax.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, marker='.')

            prev_centroids = self.centroids.copy()
            ax.scatter(prev_centroids[:, 0], prev_centroids[:, 1], c=list(range(self.n_clusters)), marker='+', cmap='winter')

            for i in range(len(self.data)):
                self.labels[i] = self._get_closest_class_id(self.data[i])

            for k in range(self.n_clusters):
                cluster_data = self.data[np.where(self.labels == k)]
                if len(cluster_data) == 0:
                    self.centroids[k] = self.data[np.random.randint(0, len(self.data))]
                else:
                    self.centroids[k] = cluster_data.mean(axis=0)

            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c=list(range(self.n_clusters)), marker='x', cmap='winter')

            # Check for convergence
            centroid_shift = np.linalg.norm(self.centroids - prev_centroids, axis=1).max()
            if centroid_shift < 1e-4:  # Convergence threshold
                anim.event_source.stop()
                # Calculate and display silhouette score
                silhouette_avg = silhouette_score(self.data, self.labels)
                ax.set_title(f'Silhouette Score: {silhouette_avg:.2f}')

    visualizer = KMeansVisualizer(n_clusters, normalized_data, algorithm, init, max_iter)
    anim = FuncAnimation(fig, visualizer.fit, frames=max_iter, interval=1000, repeat=False)
    plt.show()


def average_silhouette_score(data, n_clusters, algorithm, init="random", iterations=10, noise_level=0.0):
    scores = []
    max_score = -1

    for _ in range(iterations):
        labels = perform_kmeans(data, n_clusters, algorithm, init, noise_level=noise_level)
        score = silhouette_score(data[["AREA CODE"]], labels)
        scores.append(score)
        if score > max_score:
            max_score = score
    avg_score = np.mean(scores)
    return avg_score, max_score


# Ensure the correct column name is used for mapping crime types
def apply_fuzzy_and_label_encoder(df, column_name, fuzzy_column_name, encoder_column_name):
    # Apply fuzzy matching and group similar area names
    location_mapping_result = fuzzy_group(df[column_name].tolist())
    df[fuzzy_column_name] = df[column_name].map(location_mapping_result)

    # Convert the mapped area names to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    df[encoder_column_name] = label_encoder.fit_transform(df[fuzzy_column_name])

    return df


def process_cluster(df, cluster_label):
    cluster_df = df[df["cluster"] == cluster_label].copy()

    # Apply fuzzy matching and label encoding on "Crm Cd Desc"
    cluster_df = apply_fuzzy_and_label_encoder(cluster_df, "Crm Cd Desc", "MAPPED CRIME TYPE", "CRIME CODE")

    # Find optimal number of clusters using CRIME CODE
    optimal_k, labels = find_optimal_clusters(cluster_df[["CRIME CODE"]], plot=False)
    print(f"Optimal number of clusters for cluster {cluster_label}: {optimal_k}")

    # Ensure labels are not None
    if labels is None:
        print(f"Error: No labels generated for cluster {cluster_label}.")
    else:
        cluster_df["crime_cluster"] = labels

        # Retrieve the mapped area name for the current cluster
        location = cluster_df["MAPPED AREA NAME"].iloc[0]

        # Create a folium map centered around the mean latitude and longitude of the cluster
        mean_lat = cluster_df["LAT"].mean()
        mean_lon = cluster_df["LON"].mean()
        crime_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

        rand_colors = [
            "red", "blue", "green", "purple", "orange", "darkred", "lightred",
            "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple",
            "pink", "lightblue", "lightgreen", "gray", "black", "lightgray"
        ]
        # Generate a unique color for each crime type
        unique_crimes = cluster_df["MAPPED CRIME TYPE"].unique()
        crime_colors = {crime: random.choice(rand_colors) for crime in unique_crimes}

        # Add markers for each crime type in the cluster
        for _, row in cluster_df.iterrows():
            mapped_crime_type = row["MAPPED CRIME TYPE"]
            crime_color = crime_colors[mapped_crime_type]
            folium.Marker(
                location=[row["LAT"], row["LON"]],
                popup=(
                    f"<b>Crime Type:</b> {mapped_crime_type}<br><br>"
                    f"<b>Date Reported:</b> {row['Date Rptd']}<br><br>"
                    f"<b>Date Occurred:</b> {row['DATE OCC']}<br><br>"
                    f"<b>Time Occurred:</b> {row['TIME OCC']}<br><br>"
                    f"<b>Area Name:</b> {row['AREA NAME']}<br><br>"
                    f"<b>Victim Age:</b> {row['Vict Age']}<br><br>"
                    f"<b>Victim Sex:</b> {row['Vict Sex']}<br><br>"
                    f"<b>Status:</b> {row['Status']}<br><br>"
                    f"<b>Location:</b> {row['LOCATION']}"
                ),
                icon=folium.Icon(color=crime_color, icon="info-sign")
            ).add_to(crime_map)

        # Add a legend to the map
        crime_counts = cluster_df["MAPPED CRIME TYPE"].value_counts()
        legend_html = '''
        <div style="position: fixed; top: 10px; right: 10px; width: 200px; height: auto; max-height: 300px; overflow-y: auto; z-index:9999; font-size:14px; background-color: white; padding: 10px; border: 2px solid black; box-sizing: border-box;">
        '''
        legend_html += f'<b>Different Crimes in {location}</b><br><br>'
        for crime, count in crime_counts.items():
            color = crime_colors[crime]
            legend_html += f'<i style="background:{color};width:12px;height:12px;display:inline-block;"></i> <b style="font-size:12px;">{crime.title()}</b>: <span style="font-size:12px;">{count}</span><br><br>'
        legend_html += '</div>'
        crime_map.get_root().html.add_child(folium.Element(legend_html))

        # Save the map to an HTML file
        map_file = f'{location}_crime_map.html'
        crime_map.save(map_file)

        # Open the HTML file in the default web browser
        webbrowser.open(map_file)


df = pd.read_csv("data/Crime_Data_from_2020_to_Present.csv")
df = df[:100]

df = apply_fuzzy_and_label_encoder(df, "AREA NAME", "MAPPED AREA NAME", "AREA CODE")


# Find optimal number of clusters using AREA CODE
optimal_k, labels = find_optimal_clusters(df[["AREA CODE"]], plot=True)
print(f"Optimal number of clusters: {optimal_k}")

# Ensure labels are not None
if labels is None:
    print("Error: No labels generated.")
else:
    df["cluster"] = labels


# Visualize KMeans with Lloyd's algorithm
visualize_kmeans(df, n_clusters=len(df['cluster'].unique()), algorithm="macqueen", init="adaptive", max_iter=300, noise_level=0.0)

#     # Print each cluster's data
#     for cluster_label in df["cluster"].unique():
#         cluster_df = df[df["cluster"] == cluster_label][["AREA NAME", "MAPPED AREA NAME", "AREA CODE"]]
#         print(f"\nCluster {cluster_label}:\n", cluster_df)
#     print(f"Total number of unique clusters: {len(df['cluster'].unique())}")


# noise_level = 0.1  # Adjust the noise level as needed
#
# # Calculate average silhouette scores for different algorithms
# avg_silhouette_lloyd, max_silhouette_lloyd = average_silhouette_score(df, n_clusters=len(df['cluster'].unique()), algorithm="lloyd", noise_level=noise_level)
# avg_silhouette_elkan, max_silhouette_elkan = average_silhouette_score(df, n_clusters=len(df['cluster'].unique()), algorithm="elkan", noise_level=noise_level)
# avg_silhouette_macqueen, max_silhouette_macqueen = average_silhouette_score(df, n_clusters=len(df['cluster'].unique()), algorithm="macqueen", noise_level=noise_level)
# avg_silhouette_macqueen_adaptive, max_silhouette_macqueen_adaptive = average_silhouette_score(df, n_clusters=len(df['cluster'].unique()), algorithm="macqueen", init="adaptive", noise_level=noise_level)
#
# print(f"Average Silhouette Score - Lloyd's Algorithm (KMeans): {avg_silhouette_lloyd}")
# print(f"Average Silhouette Score - Elkan's Algorithm (KMeans): {avg_silhouette_elkan}")
# print(f"Average Silhouette Score - Macqueen's Algorithm (KMeans): {avg_silhouette_macqueen}")
# print(f"Average Silhouette Score - Adaptive Macqueen's Algorithm (KMeans): {avg_silhouette_macqueen_adaptive}")
#
# # Plot average silhouette scores
# avg_silhouette_scores = [avg_silhouette_lloyd, avg_silhouette_elkan,avg_silhouette_macqueen, avg_silhouette_macqueen_adaptive]
# algorithms = ["Lloyd's Algorithm", "Elkan's Algorithm", "Macqueen's Algorithm", "Adaptive Macqueen's Algorithm"]
# plt.bar(algorithms, avg_silhouette_scores)
# plt.ylabel("Average Silhouette Score")
# plt.title("Average Silhouette Scores of Different Clustering Algorithms")
# plt.show()
#
#
# # Scatter plot using LAT_ORIG and LON_ORIG with MAPPED AREA NAME as labels
# plt.figure(figsize=(10, 6))
# for cluster_label in df["cluster"].unique():
#     cluster_df = df[df["cluster"] == cluster_label]
#     mapped_area_name = cluster_df["MAPPED AREA NAME"].iloc[0]  # Get the mapped area name for the cluster
#     plt.scatter(cluster_df["LAT"], cluster_df["LON"], label=mapped_area_name)
#
# plt.xlabel("Latitude")
# plt.ylabel("Longitude")
# plt.title("Scatter Plot of Clusters using Latitude and Longitude")
# plt.legend()
# plt.show()
#
#
# Iterate through each unique cluster and process
for cluster_label in df["cluster"].unique():
    process_cluster(df, cluster_label)
