import ast
import random
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering, MeanShift, MiniBatchKMeans, \
    estimate_bandwidth, k_means_
from .clusterer import ClusterPaper, ClusterSlot, Clusterer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

__author__ = 'Tadej'


class ClustererNew(Clusterer):
    """
    TODO - Take into account graph data
         - Take into account locked and unlocked papers
         - Add an option to lock entire slots?
    :type papers: list[ClusterPaper]
    :type data_list : list[list[string]]
    :type slots: list[ClusterSlot]
    """

    def __init__(self, papers: list, schedule: list, schedule_settings: list, paper_schedule: list, func,
                 input_matrices: list):
        super().__init__(papers, schedule, schedule_settings, paper_schedule, func)
        self.input_matrices = input_matrices

    def create_dataset(self):
        input_matrices = self.input_matrices
        X = None
        for matrix in input_matrices:
            matrix = normalize(matrix, axis=1, norm='l1')
            if X == None:
                X = matrix
            else:
                X = np.hstack((X, matrix))
        self.data = X
        if self.func == "msh":
            try:
                band = estimate_bandwidth(self.data)
            except ValueError:
                return False
            band = band * (self.bandwith_factor / 100)
            if band == 0:
                return False
            self.cluster_function = MeanShift(bandwidth=band)
        return True

    def basic_clustering(self):
        data = self.data
        cluster_values = self.cluster_function.fit_predict(data).tolist()
        if self.func == 'kmm' or self.func == 'kme':
            cluster_distances = self.cluster_function.fit_transform(data)
        else:
            cluster_distances = []
            for paper in self.papers:
                d = []
                for a in range(0, len(cluster_values)):
                    d.append(0)
                cluster_distances.append(d)
        if self.using_dbs:
            # print("USING DBS")
            ok_clusters = cluster_values.count(1)
            if ok_clusters == 0:
                self.eps += 0.002
                self.cluster_function = DBSCAN(eps=self.eps, min_samples=5)
                if self.eps > 1:
                    print("DBS ERROR")
                    return
                return self.basic_clustering()
        for index, distance in enumerate(cluster_distances):
            self.papers[index].cluster_distances = distance
        for index, value in enumerate(cluster_values):
            self.papers[index].cluster = value
        # Assign basic clusters to papers
        if self.first_clustering == True:
            # print("first clustering")
            for index, paper in enumerate(self.papers):
                print("assigned ", paper.cluster, "to paper ", paper.paper.title)
                print(paper.cluster)
                paper.paper.simple_cluster = paper.cluster + 1
                paper.paper.simple_visual_x = self.data[index][0]
                paper.paper.simple_visual_y = self.data[index][1]
                paper.paper.visual_x = self.data[index][0]
                paper.paper.visual_y = self.data[index][1]
                paper.paper.save()
                self.first_clustering = False
        # Get coordinates for visualization
        self.get_coords()
        # for i in range(0,len(self.cluster_distances)):
        #    print(self.papers[i].title, "-", self.cluster_distances[i])
        for index, paper in enumerate(self.papers):
            pass
            # print(self.num_clusters, " assigned ", paper.cluster, "to paper " ,paper.paper.title)

    def find_papers_with_sum(self, set, subset, desired_sum, curr_index, result):
        # return after finding 10 combinations
        if len(result) >= 10:
            return
        lens = [p.paper.length for p in subset]
        if sum(lens) == desired_sum:
            indexes = []
            for paper in subset:
                indexes.append(self.papers.index(paper))
            result.append(indexes)
            return
        if sum(lens) > desired_sum:
            return
        for i in range(curr_index, len(set)):
            self.find_papers_with_sum(set, subset + [set[i]], desired_sum, i + 1, result)

    def get_coords(self):
        if (len(self.papers) == 1):
            visual_coords_x = [0]
            visual_coords_y = [0]
        else:
            self.create_dataset()
            # print("--------------COORDS------------")
            # print(pca_data[:,0])
            # print(pca_data[:,1])
            visual_coords_x = self.data[:, 0]
            visual_coords_y = self.data[:, 1]
        for index, x in enumerate(visual_coords_x):
            # self.papers[index].paper.visual_x = x
            self.papers[index].paper.save()
        for index, y in enumerate(visual_coords_y):
            # self.papers[index].paper.visual_y = y
            self.papers[index].paper.save()

    def reset_papers(self):
        for paper in self.papers:
            paper.paper.add_to_day = -1
            paper.paper.add_to_row = -1
            paper.paper.add_to_col = -1
            paper.paper.cluster = 0
            paper.paper.simple_cluster = 0
            paper.paper.save()

    def fit_to_schedule2(self):
        # print("started fitting")
        """
        An alternate approach approach:
            1.) Perform clustering with basic_clustering - done manually in the view
            2.) Select a cluster with papers most similar to one another and fill a slot
            3.) Fill slots in the following order: single slots, parallel slots
            4.) With parallel slots, each slot should be filled with papers from a different cluster - prefferebly the
                cluster centroids of such clusters should be as far away as possible
            5.) Repeat untill all slots are filled
        """
        # The following is repeated for every cluster independently
        # Slot lengths are initialized in __init__
        previous_clusters = []  # Used when picking clusters for parallel slots
        slots_to_delete = []
        # First fill out slots with locked papers by filling them with similar papers
        # Since there is no guarantee multiple locked papers will be from the same cluster only look at one of them
        for locked_slot in self.locked_slots:
            # Find first locked paper of the slot - all non locked are already removed at this point
            day = locked_slot.coords[0]
            row = locked_slot.coords[1]
            col = locked_slot.coords[2]
            locked_paper_id = self.paper_schedule[day][row][col][0]
            # Find ths cluster containing the paper
            containing_cluster = 0
            for paper in self.papers:
                if paper.paper.submission_id == locked_paper_id:
                    containing_cluster = paper.cluster
            # The length of papers in the slot must be subtracted from the slot length
            for paper_id in self.paper_schedule[day][row][col]:
                for paper in self.papers:
                    if paper.paper.submission_id == paper_id:
                        locked_slot.length -= paper.paper.length
            # Fit slot with papers from the same cluster, if able
            cluster_papers = [p for p in self.papers if p.cluster == containing_cluster]
            papers = []
            self.find_papers_with_sum(cluster_papers, [], locked_slot.length, 0, papers)
            # If the cluster does not contain enough papers, try with all papers
            if len(papers) == 0:
                self.find_papers_with_sum(self.papers, [], locked_slot.length, 0, papers)
                # If a valid combination cannut be found even with this, the slot cannot be fitted
            if len(papers) == 0:
                continue
            # if there are multiple fitting groups in the same cluster, select the group with the smallest error
            selected_index = 0
            if len(papers) > 1:
                min_error = 9999999999999
                for index, subset in enumerate(papers):
                    error = 0
                    for paper in subset:
                        error += self.papers[paper].cluster_distances[containing_cluster] * \
                                 self.papers[paper].cluster_distances[containing_cluster]
                    if error < min_error:
                        selected_index = index
                        min_error = error
            print(str(len(self.papers)))
            print(str(len(papers)))

            # ids = [self.papers[i].paper.id for i in papers[selected_index]]
            papers_to_update = [(self.papers[i], i) for i in papers[selected_index]]

            # print("PAPERS TO UPATE: ", papers_to_update)
            # Return the cluster coordinates - used for visualization
            for paper, index in papers_to_update:
                paper.paper.cluster = self.current_cluster
                coords = locked_slot.coords
                paper.paper.add_to_day = coords[0]
                paper.paper.add_to_row = coords[1]
                paper.paper.add_to_col = coords[2]
                # print("COORD ",  index, self.visual_coords_x[index], self.visual_coords_y[index])
                paper.paper.save()
            self.current_cluster += 1
            # remove the assigned papers from this class, since they no longer need to be assigned
            offset = 0
            for paper, index in papers_to_update:
                self.papers.remove(paper)
                # Remove the relevant line from graph dataset
                if self.using_graph_data == True:
                    graph = len(self.graph_dataset)
                    del self.graph_dataset[index - offset]
                    # since indexes are sorted in asending order, they must be updated, which is handled by offset
                    offset += 1
            # also remove the information about the slot
            del locked_slot
            # delete marked slot
            # print("SLOTS TO DELETE: ", slots_to_delete, len(self.slots))
            if len(slots_to_delete) > 0:
                del self.slots[slots_to_delete[0]]
                del slots_to_delete[0]

        while self.slots != []:
            # print("slots")
            # for slot in self.slots:
            # print(slot.length, slot.is_parallel, slot.sub_slots)
            do_break = False
            # Get biggest empty slot - only select parallel slots once all non-parallel slots have already been filled
            slot_length = 0
            slot_index = 0
            sub_slot = None
            num_nonparallel = 0
            for slot in self.slots:
                if slot.is_parallel == False:
                    num_nonparallel += 1
            for index, slot in enumerate(self.slots):
                if slot.length > slot_length:
                    # Skip parallel slots until there are no other options
                    if num_nonparallel > 0 and slot.is_parallel == True:
                        continue
                    if slot.is_parallel:
                        # Ignore empty slots - they will be deleted later
                        if (len(slot.sub_slots) == 0):
                            previous_clusters = []
                            slots_to_delete.append(index)
                            continue
                        # For parallel slots, pick the first unfilled subslot. If all subslots have been filled,
                        # delete the slot and continue
                        else:
                            sub_slot = slot.sub_slots[0]
                            slot_length = sub_slot.length
                    else:  # slot is not parallel
                        slot_length = slot.length
                    slot_index = index
            # If slot is not parallel, select a single cluster
            if slot_length == 0:
                break
            if not self.slots[slot_index].is_parallel:
                # Select biggest cluster
                cluster_values = [paper.cluster for paper in self.papers]
                max_cluster_index = max(cluster_values)
                # print("values ", cluster_values)
                cluster_sizes = [cluster_values.count(i) for i in range(0, max_cluster_index + 1)]
                max_cluster = cluster_sizes.index(max(cluster_sizes))
                # Get papers from that cluster
                cluster_papers = [p for p in self.papers if p.cluster == max_cluster]
            else:
                # If the slot is parallel, then consider previous clusters
                cluster_values = [paper.cluster for paper in self.papers]
                cluster_sizes = [cluster_values.count(i) for i in range(0, len(self.papers))]
                max_size = 0
                max_cluster = None
                for index, size in enumerate(cluster_sizes):
                    if index in previous_clusters:
                        # print("PREVIOUS CLUSTER")
                        continue
                    if size > max_size:
                        max_size = size
                        max_cluster = index
                if max_cluster == None:
                    max_cluster = cluster_sizes.index(max(cluster_sizes))
                    # This means that there is not enough clusters to ignore previous clusters, so we don't
                    pass
                cluster_papers = [p for p in self.papers if p.cluster == max_cluster]
            # print(cluster_sizes)
            # print("SELECTED ", max_cluster, " with size", len(cluster_papers))
            # If slot is parallel, then the previous clusters must also be considered - simultaneous parallel slots should
            # be filled whith papers from different clusters
            # Select papers that fit into the slot
            papers = []
            # print("finding papers")
            self.find_papers_with_sum(cluster_papers, [], slot_length, 0, papers)
            # print("found papers")
            # print(papers)
            if papers == []:
                # This happens when there are no papers, that can completely fill a slot in the largest cluster.
                # In this case, it makes sense to rerun clustering with less clusters, as that should produce clusters
                #   with more papers.
                # If even that doesnt help, then the function should end end report this to the user
                if self.func == "msh":
                    cond = (self.bandwith_factor >= 300)
                if self.func == "hie" or self.func == "kmm" or self.func == "kme":
                    cond = (self.num_clusters == 0)
                if self.func == "aff":
                    # print("starting merge")
                    merged = []
                    # This one is problematic - manually merge clusters
                    cond = False
                    centers = self.cluster_function.cluster_centers_
                    self.clusters_merged += 1
                    if self.clusters_merged >= 20:
                        print("failed to cluster")
                        return False
                    for x in range(self.clusters_merged):
                        # Merge 2 nearest clusters
                        min_dist = 1000000
                        cluster1 = 0
                        cluster2 = 0
                        for i in range(len(centers)):
                            coord1 = np.array(centers[i])
                            if i in merged:
                                continue
                            for j in range(i + 1, len(centers)):
                                if j in merged:
                                    continue
                                coord2 = np.array(centers[j])
                                if np.linalg.norm(coord1 - coord2) < min_dist:
                                    cluster1 = i
                                    cluster2 = j
                                    min_dist = np.linalg.norm(coord1 - coord2)
                        # print("merged ", cluster1, cluster2)
                        merged.append(cluster1)
                        # Change paper clusters
                        for paper in self.papers:
                            if paper.cluster == cluster1:
                                paper.cluster = cluster2
                    return self.fit_to_schedule2()
                if cond:
                    print("failed to cluster")
                    return False
                else:
                    # Needed, since clustering cannot be performed if n_samples < n_clusters
                    self.num_clusters -= 1
                    if self.num_clusters == 0 and (self.func == "hie" or self.func == "kmm" or self.func == "kme"):
                        print("failed to cluster")
                        return False
                    self.bandwith_factor += 10
                    self.set_cluster_function(self.func)
                    if not self.create_dataset():
                        print("failed to cluster")
                        return False
                    self.basic_clustering()
                    # print("NO SUITABLE COMBINATION FOUND2")
                    return self.fit_to_schedule2()
            else:
                # if there are multiple fitting groups in the same cluster, select the group with the smallest error
                selected_index = 0
                if len(papers) > 1:
                    min_error = 9999999999999
                    for index, subset in enumerate(papers):
                        error = 0
                        for paper in subset:
                            error += self.papers[paper].cluster_distances[max_cluster] * \
                                     self.papers[paper].cluster_distances[max_cluster]
                        if error < min_error:
                            selected_index = index
                            min_error = error
            # Update the papers' add_to_day/row/col fields. This fields will then be used to add the papers into the schedule
            # Also update the papers' cluster field
            ids = [self.papers[i].paper.id for i in papers[selected_index]]
            papers_to_update = [(self.papers[i], i) for i in papers[selected_index]]
            # print("PAPERS TO UPATE: ", papers_to_update)
            # Return the cluster coordinates - used for visualization
            for paper, index in papers_to_update:
                paper.paper.cluster = self.current_cluster
                if not self.slots[slot_index].is_parallel:
                    coords = self.slots[slot_index].coords
                else:
                    coords = sub_slot.coords
                print('added 1', coords[0], coords[1], coords[2])
                paper.paper.add_to_day = coords[0]
                paper.paper.add_to_row = coords[1]
                paper.paper.add_to_col = coords[2]
                # print("COORD ",  index, self.visual_coords_x[index], self.visual_coords_y[index])
                paper.paper.save()
            self.current_cluster += 1
            # remove the assigned papers from this class, since they no longer need to be assigned
            offset = 0
            for paper, index in papers_to_update:
                self.papers.remove(paper)
                # Remove the relevant line from graph dataset
                if self.using_graph_data == True:
                    graph = len(self.graph_dataset)
                    del self.graph_dataset[index - offset]
                    # since indexes are sorted in asending order, they must be updated, which is handled by offset
                    offset += 1
            # also remove the information about the slot
            if not self.slots[slot_index].is_parallel:
                del self.slots[slot_index]
            else:
                previous_clusters.append(max_cluster)
                del self.slots[slot_index].sub_slots[0]

            # delete marked slot
            # print("SLOTS TO DELETE: ", slots_to_delete, len(self.slots))
            if len(slots_to_delete) > 0:
                del self.slots[slots_to_delete[0]]
                del slots_to_delete[0]
                # redo clustering
                # self.set_cluster_function(self.cluster_function)
                # self.create_dataset()
                # self.basic_clustering()

    def constrained_clustering(self, X, n_iters=1):
        print('running with X', np.array(X).shape)
        for line in X:
            print(line)
        # First calculate the number of needed clusters and their sizes
        # Clusters is the list where each element is a list of lengths in a parallel slot. If a slot is not
        # parallel then it's a list of length 1
        results = []
        for i in range(n_iters):
            print('DOING CONSTRAINED CLUSTERING')
            clusters = []
            clusters_lengths = []
            num_clusters = 0
            for d, day in enumerate(self.schedule):
                for r, row in enumerate(day):
                    row_clusters = []
                    for c, col in enumerate(row):
                        print('added to clusters:', self.schedule_settings[d][r][c])
                        row_clusters.append(self.schedule_settings[d][r][c])
                        num_clusters += 1
                        clusters_lengths.append(self.schedule_settings[d][r][c])
                    clusters.append(row_clusters)
            print('clusters:', clusters, num_clusters)
            # Then obtain centers with k-means++
            # This can be done with sklearn.KMeans fit transform, which computes the clutsers and returns
            # distances of points to cluster centers
            print('initial clustering for centroids')
            print('number of clusters', num_clusters)
            clusterer = KMeans(n_clusters=num_clusters, init='k-means++')
            cluster_centers = k_means_._init_centroids(np.array(X), num_clusters, 'k-means++')
            print('cluster_centers shape', np.array(cluster_centers).shape)
            # Compute distances to centers for each point
            cluster_distances = []
            for point in X:
                point_distances = []
                for center in cluster_centers:
                    point_distances.append(np.linalg.norm(point - center))
                cluster_distances.append(point_distances)
            cluster_distances = np.array(cluster_distances)
            # cluster_distances = clusterer.fit_transform(X)
            print('found distances', cluster_distances)
            # cluster_centers = clusterer.cluster_centers_
            # We want to assign points to clusters in a decently optimal way
            # This is done by sorting points based on distance_to_nearest_cluster - distance_to_furthest_cluster
            # Distance is euclidian distance - ie. sum of squares. This is also the one used in Kmeans and sklearn
            min_dists = []
            max_dists = []
            closest_clusters = []
            point_indices = []
            for i, point_distances in enumerate(cluster_distances):
                min_dist = np.finfo(np.float64).max
                max_dist = 0
                min_cluster = -1
                for id, distance in enumerate(point_distances):
                    if distance < min_dist:
                        min_dist = distance
                        min_cluster = id
                    if distance > max_dist:
                        max_dist = distance
                min_dists.append(min_dist)
                max_dists.append(max_dist)
                closest_clusters.append(min_cluster)
                point_indices.append(i)
                print('dists were', i, min_dist, max_dist)
            # Now sort the points based on these distances
            sorted_points = sorted(point_indices, key=lambda x: min_dists[x] - max_dists[x])
            for i in sorted_points:
                print(i, min_dists[i], max_dists[i])
            print('doing initial assignments')
            # Assign those points to their closest clusters in order. If that cluster is full than assign them to the
            # next closest cluster. The minimum cluster was kept before as a small optimization.
            cluster_assignments = {}
            clusters_of_points = np.zeros(len(point_indices), dtype=int)
            for point_index in sorted_points:
                closest_cluster = closest_clusters[point_index]
                # If the cluster is empty adding is not a problem - we assume an empty slot will always have enough length
                # to take in atleast one paper.
                if closest_cluster not in cluster_assignments.keys():
                    cluster_assignments[closest_cluster] = [point_index]
                    clusters_of_points[point_index] = closest_cluster
                # IF the cluster is not empty check if it is full
                else:
                    # TODO - *20 is temporary. Currently all papers are 20 minutes long so this works. This will probably
                    # be changed so that slots have a length in actual papers instead of imultes
                    # If the cluster isn't full we can add the point
                    if (len(cluster_assignments[closest_cluster]) * 20) < clusters_lengths[closest_cluster]:
                        cluster_assignments[closest_cluster].append(point_index)
                        clusters_of_points[point_index] = closest_cluster
                    # If it is full then sort the clusters by distance and keep trying until finding a non empty one
                    else:
                        point_distances = cluster_distances[point_index]
                        sorted_indices = [d[0] for d in sorted(enumerate(point_distances), key=lambda x: x[1])]
                        for i in sorted_indices:
                            # Still need to check if we found an empty cluster
                            if i not in cluster_assignments.keys():
                                cluster_assignments[i] = [point_index]
                                clusters_of_points[point_index] = i
                                break
                            elif (len(cluster_assignments[i]) * 20) < clusters_lengths[i]:
                                cluster_assignments[i].append(point_index)
                                clusters_of_points[point_index] = i
                                break
            print('cluster assignments', cluster_assignments)
            """
            This was just the initialization
            Since K-means is not really meant for this kind of constrained clustering it is now helpful to do some
            local optimization
            This is done in the following way:
                1.) Calculate cluster centers
                2.) For each point, calculate distances to new centers
                3.) Find the best possible alternate cluster and sort points based on improvement
                4.) Try to swap each element in order by an element from another cluster, if this swap yields an improvement
            """
            swap_occured = True
            while swap_occured:
                swap_occured = False
                # Calculate cluster centers
                for key, item in cluster_assignments.items():
                    cluster_points = []
                    for point in item:
                        # print(np.array(X[point]).shape)
                        cluster_points.append(X[point])
                    cluster_points = np.array(cluster_points)
                    # print('cluster points', cluster_points.shape)
                    avg = np.average(cluster_points, axis=0)
                    # print('avg', avg)
                    cluster_centers[key] = avg
                # Compute distances to new centers for each point
                new_distances = []
                for point in X:
                    point_distances = []
                    for center in cluster_centers:
                        point_distances.append(np.linalg.norm(point - center))
                    new_distances.append(point_distances)
                # Find better alternate assignments
                # Larger difference means a higher improvement
                differences = []
                for i, point in enumerate(X):
                    point_differences = []
                    curr_dist = new_distances[i][clusters_of_points[i]]
                    for distance in new_distances[i]:
                        diff = curr_dist - distance
                        point_differences.append(diff)
                    differences.append(point_differences)
                differences = np.array(differences)
                # Select the element with the largest difference and attempt to swap
                # The swap is only possible if the sum of both distances to centres after the swap is smaller
                # If it is larger proceed with the next highest difference
                # Only swap each element once, if more is needed that will be done in the next iteration
                already_swapped_elements = set()
                partition = np.argpartition(differences.flatten(), -1000)
                numswaps = 0
                for i in range(1, 1000):
                    if len(already_swapped_elements) >= len(X):
                        break
                    max = partition[-i]
                    # max_index = np.unravel_index(differences.argmax(), differences.shape)
                    max_index = np.unravel_index(max, differences.shape)
                    # print('max index', max_index)
                    point1i = max_index[0]
                    point1_cluster = clusters_of_points[point1i]
                    point2_cluster = max_index[1]
                    # select points actually in point2_cluster
                    possible_points2 = cluster_assignments[point2_cluster]
                    distance_to_cluster2 = []
                    for p in possible_points2:
                        distance_to_cluster2.append(new_distances[p][point2_cluster])
                    point2i = possible_points2[np.argmax(distance_to_cluster2)]
                    # select the one that has the highest distance from the centroid
                    # print('shape1', (differences[point1_cluster, :]).shape) this has classes
                    # print('shape2', (differences[:, point1_cluster]).shape) this has points
                    # point2i = (differences[point1_cluster, :]).argmax()
                    if point1i in already_swapped_elements or point2i in already_swapped_elements:
                        continue
                    p1c = differences[:, point1_cluster]
                    p2c = differences[:, point2_cluster]
                    # print('point1c', p1c)
                    # print('point2c', p2c)
                    # print(point1i, point1_cluster, point2i, point2_cluster)
                    dist_1_pre_swap = new_distances[point1i][point1_cluster]
                    dist_2_pre_swap = new_distances[point2i][point2_cluster]
                    dist_1_post_swap = new_distances[point1i][point2_cluster]
                    dist_2_post_swap = new_distances[point2i][point1_cluster]
                    if dist_1_pre_swap + dist_2_pre_swap <= dist_1_post_swap + dist_2_post_swap:
                        continue
                    if point1_cluster == point2_cluster:
                        continue
                    # print('dist pre swap', dist_1_pre_swap + dist_2_pre_swap, dist_1_pre_swap, dist_2_pre_swap)
                    # print('dist post swap', dist_1_post_swap + dist_2_post_swap, dist_1_post_swap, dist_2_post_swap)
                    # To ensure each element is only swapped once remember them
                    already_swapped_elements.add(point1i)
                    already_swapped_elements.add(point2i)
                    numswaps += 1
                    # Make the swap
                    print('p1', point1i, cluster_assignments[point1_cluster])
                    print('p2', point2i, cluster_assignments[point2_cluster])
                    print('dists', dist_1_pre_swap + dist_2_pre_swap, dist_1_post_swap + dist_2_post_swap)
                    total = 0
                    for key, item in cluster_assignments.items():
                        for _ in item:
                            total += 1
                    print('cluster_assignments', total, 'points', len(X))
                    cluster_assignments[point1_cluster].remove(point1i)
                    cluster_assignments[point2_cluster].remove(point2i)
                    cluster_assignments[point2_cluster].append(point1i)
                    cluster_assignments[point1_cluster].append(point2i)
                    clusters_of_points[point1i] = point2_cluster
                    clusters_of_points[point2i] = point1_cluster
                    # print('len', len(already_swapped_elements))
                    swap_occured = True
                print('numswaps', numswaps)
            # Compute average distances within cluster
            sum_distances = {}
            for key, item in cluster_assignments.items():
                sum_dist = []
                for paper1i in range(len(item)):
                    for paper2i in range(paper1i + 1, len(item)):
                        p1 = X[item[paper1i]]
                        p2 = X[item[paper2i]]
                        sum_dist.append(np.linalg.norm(p1 - p2))
                sum_distances[key] = sum_dist
            results.append((cluster_assignments, sum_distances, cluster_centers))
        # End of iters loop
        # Pick the best result
        sums = []
        for assignments, s_distances, _ in results:
            sum_distances = 0
            for _, distances in s_distances.items():
                print('distances', distances)
                if len(distances) == 0:
                    sum_distances = 1000000
                else:
                    sum_distances += sum(distances) / len(distances)
            sums.append(sum_distances)
        best_i = np.argmin(np.array(sums))
        best_cluster_assignments, best_sum_distances, cluster_centers = results[best_i]
        available_keys = list(best_cluster_assignments.keys())
        print('sums were', sums)
        # Set coordinates for adding to schedule
        print('clusterer schedule', self.schedule)
        slot_centroids = []
        for d, day in enumerate(self.schedule):
            for r, row in enumerate(day):
                # pick clusters so that the ones in parallel slots are far away from each other
                # each c should have a cluster far away from the others
                current_centroids = []
                for c, col in enumerate(row):
                    print('running innitial assignment at', d, r, c)
                    # filter out clusters of incorrect lengths
                    # TODO - currently each paper is assumed to take 20 minutes. This will need to be changed.
                    slot_length = int(self.schedule_settings[d][r][c] / 20)
                    print('slot length', slot_length)
                    fitting_keys = []
                    for key in available_keys:
                        print('key length', len(best_cluster_assignments[key]), slot_length)
                        if (len(best_cluster_assignments[key])) == slot_length:
                            fitting_keys.append(key)
                            print('added')
                    print('fitting keys', fitting_keys)
                    if fitting_keys == []:
                        print('ERROR FITTING KEYS')
                        continue
                    if current_centroids == []:
                        print('entering true')
                        key = fitting_keys[0]
                        available_keys.remove(key)
                        papers = best_cluster_assignments[key]
                        current_centroids.append(cluster_centers[key])
                    # Pick something far away from current centers
                    else:
                        print('entering false', fitting_keys)
                        max_dist = 0
                        max_key = 0
                        for key in fitting_keys:
                            dist_sum = 0
                            for current_center in current_centroids:
                                dist = np.linalg.norm(cluster_centers[key] - current_center)
                                dist_sum += dist
                            if dist_sum > max_dist:
                                max_dist = dist_sum
                                max_key = key
                        available_keys.remove(max_key)
                        current_centroids.append(cluster_centers[max_key])
                        papers = best_cluster_assignments[max_key]
                        print('clusterer papers', papers, slot_length)
                        print('clusters left', len(available_keys))
                    for paper_index in papers:
                        print('paper index was', paper_index)
                        paper = self.papers[paper_index]
                        print('added 2', d, r, c,  'to index', paper_index)
                        paper.paper.add_to_day = d
                        paper.paper.add_to_row = r
                        paper.paper.add_to_col = c
                        paper.paper.save()
                        # print('culsterer', paper.paper.title, paper.paper.add_to_day, paper.paper.add_to_row, paper.paper.add_to_col)
                slot_centroids.append((current_centroids, slot_length))
        # Run optimization on parallel slots
        # Swap two slots if:
        #   They have the same size
        #   They appear in different parallel sections
        #   The swap increases the total distance from centers
        #   Calculate total distances of slots from centers for both slots before and after swap, swap if that increases
        #      the sum of both distances
        slot_swap_occured = True
        while slot_swap_occured:
            swapped_num = 0  # Purely for logging
            slot_swap_occured = False
            to_swap1 = []
            to_swap2 = []
            # Pick a slot and try to find a suitable swap
            for session_i_1, (parallel_session_centroids1, length1) in enumerate(slot_centroids):
                # Calculate center
                total_center1 = np.average(np.array(parallel_session_centroids1), axis=0)
                for center_i_1, slot_center1 in enumerate(parallel_session_centroids1):
                    for session_i_2, (parallel_session_centroids2, length2) in enumerate(slot_centroids):
                        if slot_swap_occured == True:
                            break
                        if length1 != length2:
                            continue
                        print('sessions', session_i_1, session_i_2)
                        total_center2 = np.average(np.array(parallel_session_centroids2), axis=0)
                        # print('p1', parallel_session_centroids1)
                        # print('p2', parallel_session_centroids2)
                        p1s = np.array(parallel_session_centroids1).shape
                        p2s = np.array(parallel_session_centroids2).shape
                        print('shapes', p1s, p2s)
                        if p1s != p2s:
                            continue
                        comparison = np.equal(np.array(parallel_session_centroids1).flatten(),
                                              np.array(parallel_session_centroids2).flatten())
                        if all(comparison):
                            continue
                        # print('comparison', comparison)
                        for center_i_2, slot_center2 in enumerate(parallel_session_centroids2):
                            sum_dists_11 = 0
                            sum_dists_12 = 0
                            sum_dists_21 = 0
                            sum_dists_22 = 0
                            for x in parallel_session_centroids1:
                                print('d11', np.linalg.norm(slot_center1 - x))
                                sum_dists_11 += np.linalg.norm(slot_center1 - x)
                            for x in parallel_session_centroids1:
                                if all(x == slot_center1):
                                    print('d12', 0.0)
                                    continue
                                print('d12', np.linalg.norm(slot_center2 - x))
                                sum_dists_12 += np.linalg.norm(slot_center2 - x)
                            for x in parallel_session_centroids2:
                                if all(x== slot_center2):
                                    print('d21', 0.0)
                                    continue
                                print('d21', np.linalg.norm(slot_center1 - x))
                                sum_dists_21 += np.linalg.norm(slot_center1 - x)
                            for x in parallel_session_centroids2:
                                print('d22', np.linalg.norm(slot_center2 - x))
                                sum_dists_22 += np.linalg.norm(slot_center2 - x)
                            dist11 = np.linalg.norm(total_center1 - slot_center1)
                            dist12 = np.linalg.norm(total_center1 - slot_center2)
                            dist21 = np.linalg.norm(total_center2 - slot_center1)
                            dist22 = np.linalg.norm(total_center2 - slot_center2)
                            print('dists', sum_dists_11, sum_dists_12, sum_dists_21, sum_dists_22)
                            if sum_dists_11 + sum_dists_22 < sum_dists_12 + sum_dists_21:
                                # Do swap
                                print('FOUND SWAPPABLE', sum_dists_11, sum_dists_12, sum_dists_22, sum_dists_21)
                                slot_swap_occured = True
                                swapped_num += 1
                                # Find the papers
                                d1 = -1
                                d2 = -1
                                r1 = -1
                                r2 = -1
                                c1 = -1
                                c2 = -1
                                curr_row = 0
                                for di, day in enumerate(self.schedule):
                                    for ri, row in enumerate(day):
                                        if curr_row == session_i_1:
                                            r1 = ri
                                            d1 = di
                                        if curr_row == session_i_2:
                                            r2 = ri
                                            d2 = di
                                        curr_row += 1
                                # Swap papers
                                paper1 = None
                                paper2 = None
                                to_swap1.append((session_i_1, center_i_1))
                                to_swap2.append((session_i_2, center_i_2))
                                for c in range(len(self.schedule[d1])):
                                    for paper in self.papers:
                                        if paper.paper.add_to_day == d1 and paper.paper.add_to_row == r1 and paper.paper.add_to_col == c:
                                            paper1 = paper.paper
                                            c1 = c
                                        if paper.paper.add_to_day == d2 and paper.paper.add_to_row == r2 and paper.paper.add_to_col == c:
                                            paper2 = paper.paper
                                            c2 = c
                                    print('swapping', d1, r1, c1, 'with', d2, r2, c2)
                                    paper1.add_to_day = d2
                                    paper1.add_to_row = r2
                                    paper1.add_to_col = c2
                                    paper2.add_to_day = d1
                                    paper2.add_to_row = r1
                                    paper2.add_to_col = c1
                                    # Swap centers
                                break
            #print('slot_centroids shape', np.array(slot_centroids).shape)
            #print('slot_centroids[0] shape', np.array(slot_centroids)[0].shape)
            #print('slot_centroids[1] shape', np.array(slot_centroids)[1].shape)
            #print('slot_centroids[0][0] shape', len(np.array(slot_centroids)[0][0]))
            for s1, s2 in zip(to_swap1, to_swap2):
                print('s1', s1[0], s1[1], 's2', s2[0], s2[1])
                e1 = slot_centroids[s1[0]][0][s1[1]]
                e2 = slot_centroids[s2[0]][0][s2[1]]
                slot_centroids[s1[0]][0][s1[1]] = e2
                print('e1', e1)
                print('e2', e2)
                slot_centroids[s2[0]][0][s2[1]] = e1
            print('SWAPPED NUM 2', swapped_num)
        return best_cluster_assignments, best_sum_distances

    def visualize_points(self, X, labels):

        tsne_data = TSNE(n_components=2).fit_transform(X)
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], marker='o')
        print(np.array(X).shape)
        print(np.array(tsne_data).shape)
        print(np.array(labels).shape)
        for label, x, y in zip(labels, tsne_data[:, 0], tsne_data[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.show()

    def evaluate_clustering_adjusted_rand_index(self, clusters_true, clusters_pred):
        """
        clusters_true are passed in as a list (dataset_manager.raw_data_classes)
        clusters_pred are passed in as a dictionary (clusterer_new.constrained_clustering)
        :param clusters_true: list
        :param clusters_pred: dict
        :return:
        """
        # Reverse cluster_pred dict
        reverse_clusters_pred = {}
        for cluster, papers in clusters_pred.items():
            for paper in papers:
                reverse_clusters_pred[paper] = cluster
        # Change ground truth to values
        clusters_true_numbers = []
        encountered = set()
        current_i = 0
        for x in clusters_true:
            if x in encountered:
                clusters_true_numbers.append(current_i)
            else:
                encountered.add(x)
                current_i += 1
                clusters_true_numbers.append(current_i)
        clusters_pred_list = []
        for i in range(len(clusters_true)):
            clusters_pred_list.append(reverse_clusters_pred[i])
        print('clusters_true', clusters_true)
        print('clusters_true_numbers', clusters_true_numbers)
        print('reversed_clusters_pred', reverse_clusters_pred)
        print('clusters_pred_list', clusters_pred_list)
        return adjusted_rand_score(clusters_true_numbers, clusters_pred_list)

    def evaluate_clustering_silhouette(self, X, clusters_dict, rand=False, km=False):
        # Convert clusters_dict to list
        X = np.array(X)
        reverse_clusters_pred = {}
        for cluster, papers in clusters_dict.items():
            for paper in papers:
                reverse_clusters_pred[paper] = cluster
        clusters_pred_list = []
        keys = sorted(list(reverse_clusters_pred.keys()))
        for key in keys:
            clusters_pred_list.append(reverse_clusters_pred[key])
        counts = [[x, clusters_pred_list.count(x)] for x in set(clusters_pred_list)]
        print('counts', counts)
        print('X', X)
        print('X shape', X.shape)
        if rand:
            random.shuffle(clusters_pred_list)
        if km:
            clusters_pred_list = KMeans(n_clusters=35).fit_predict(X)
            print('silh classes kmeans', clusters_pred_list, len(clusters_pred_list))
        print('silh classes', clusters_pred_list, len(clusters_pred_list))
        silh_score = silhouette_score(X, clusters_pred_list)
        print('silhouette_score', silh_score)
        return silh_score
