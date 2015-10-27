import csv
import os
from igraph.remote.gephi import GephiGraphStreamer, GephiConnection
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import igraph

try:
    __file__
except NameError:
    my_path = '/Users/test/fall_2015/p2psocnet/congress_cosponsorship'
else:
    my_path = os.path.dirname(os.path.realpath(__file__))

HOUSE_CSV_PATH = my_path + "/house_full.csv"
SENATE_CSV_PATH = my_path + "/senate_full.csv"
HOUSE_MEMBERS_FOLDER = my_path + "/cosponsorship2010/house_members"
SENATE_MEMBERS_FOLDER = my_path + "/cosponsorship2010/senate_members"
HOUSE_MATRIX_FOLDER = my_path + "/cosponsorship2010/house_matrices"
SENATE_MATRIX_FOLDER = my_path + "/cosponsorship2010/senate_matrices"
SUPPORTED_CONGRESSES = ["%03d" % x for x in range(93, 111)]


# (congress #, icpsr #) -> party #
def load_party_data(which_party = "senate"):
    if which_party == "senate":
        reader = csv.reader(open(SENATE_CSV_PATH, 'r'))
    else:
        reader = csv.reader(open(HOUSE_CSV_PATH, 'r'))
    reader.__next__()
    party_lookup = {}

    for row in reader:
        party_lookup[(int(row[0]), int(row[1]))] = int(row[5])

    return party_lookup


# (congress #, row #) -> (name, party #)
def load_members(which_chamber="senate"):
    party_lookup = load_party_data(which_chamber)
    members = {}

    for congress in SUPPORTED_CONGRESSES:
        if which_chamber == "senate":
            reader = csv.reader(open(SENATE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_senators.txt", 'r'))
        else:
            reader = csv.reader(open(HOUSE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_house.txt", 'r'))

        for i, row in enumerate(reader):
            if "NA" in row[2]:
                continue
            if (int(congress), int(row[2])) in party_lookup.keys():
                members[(int(congress), i)] = (row[0], party_lookup[(int(congress), int(row[2]))])

    return members


def _load_adjacency_matrix(bill_data):
    result = np.zeros((len(bill_data), len(bill_data)))
    for billCol in range(len(bill_data[0])):
        sponsor = None
        cosponsors = []
        for i in range(len(bill_data)):
            if int(bill_data[i][billCol]) == 1:
                sponsor = i
            if int(bill_data[i][billCol]) not in [0,1,5]: # no relation, primary sponsor, withdrawn support
                cosponsors.append(i)
        if sponsor is None:
            # print("WARNING: there is no sponsor for bill %s" % billCol)
            continue

        for cosponsor in cosponsors:
            result[cosponsor][sponsor] += 1
    return result


def load_adjacency_matrices(which_congress, which_chamber = "senate"):
    which_congress = "%03d" % int(which_congress)

    if which_chamber == "senate":
        reader = csv.reader(open(my_path + "/cosponsorship2010/senate_matrices/%s_senmatrix.txt" % which_congress, 'r'))
    else:
        reader = csv.reader(open(my_path + "/cosponsorship2010/house_matrices/%s_housematrix.txt" % which_congress, 'r'))

    bill_data = []
    for row in reader: bill_data.append(row)
    adj_matrix = _load_adjacency_matrix(bill_data)

    return adj_matrix


def get_cosponsorship_graph(congress, which_chamber="senate", return_largest_connected_only=True):
    adj_matrix = load_adjacency_matrices(congress, which_chamber)
    member_lookup = load_members(which_chamber=which_chamber)

    G = igraph.Graph()
    for i in range(len(adj_matrix)):
        if (int(congress), i) in member_lookup.keys():
            member = member_lookup[(int(congress), i)]
            G.add_vertex(i, label=member[0], party=member[1])
        else:
            G.add_vertex(i)
    G.add_vertices(list(range(0, len(adj_matrix))))  # setup our graph

    for row in range(len(adj_matrix)):
        for col in range(row, len(adj_matrix)):
            if adj_matrix[row][col] != 0:
                G.add_edge(row, col, weight=adj_matrix[row][col])

    if return_largest_connected_only:
        components = G.components()
        # find the largest connected component
        largest_connected_component = max(components, key=lambda x: len(x))
        # print(largest_connected_component)
        G = components.subgraph(list(components).index(largest_connected_component))

    return G


def get_cosponsorship_graph_nx(congress, which_chamber="senate", return_largest_connected_only=True):
    adj_matrix = load_adjacency_matrices(congress, which_chamber)

    G = nx.Graph()
    for i in range(0, len(adj_matrix)): G.add_node(i)  # setup our graph

    for row in range(len(adj_matrix)):
        for col in range(row, len(adj_matrix)):
            if adj_matrix[row][col] != 0:
                G.add_edge(row, col, weight=adj_matrix[row][col])

    if return_largest_connected_only:
        G = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]

    return G


def output_to_gexf(G, congress, which_house):
    file = open(congress + "_" + which_house + "_cosponsorship.gexf", 'w')
    file.write("""<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
    <meta lastmodifieddate="2009-03-20">
        <creator>Tom Werner</creator>
        <description>%s %s cosponsorship network</description>
    </meta>
    <graph mode="static" defaultedgetype="directed">\n""" % (congress, which_house))
    file.write("\t\t<nodes>\n")
    for node in G.vs:
        file.write('\t\t\t<node id="%d" label="%s" party="%s"/>\n' % (node.index, node['label'], node['party']))
    file.write("\t\t</nodes>\n")

    file.write("\t\t<edges>\n")
    for edge in G.es:
        file.write('<edge id="%d" source="%d" target="%d" weighted="%d" />\n' % (edge.index, edge.source, edge.target, edge['weight']))
    file.write("\t\t</edges>\n")
    file.write("""
    </graph>
</gexf>\n""")




    file.close()


def detect_communities(chamber="senate"):
    chamber_members = load_members(which_chamber=chamber)

    for congress in SUPPORTED_CONGRESSES:
        print("-" * 80 + "\nCongress: " + congress)
        G = get_cosponsorship_graph(congress, which_chamber=chamber)
        output_to_gexf(G, congress, chamber)

        coms = G.community_walktrap(weights=G.es['weight'])
        print("Optimal modularity:", coms.optimal_count)
        clusters = coms.as_clustering(coms.optimal_count)
        for cluster in [cluster for cluster in clusters if len(cluster) > 10]:
            members = [chamber_members[(int(congress), x)] for x in cluster if
                       (int(congress), x) in chamber_members.keys()]
            print("-" * 40)
            print("Dems:", len([temp for temp in members if temp[1] == 100]))
            print("Reps:", len([temp for temp in members if temp[1] == 200]))
        print(len([cluster for cluster in clusters if len(cluster) <= 10]), "clusters with less than 10 people.")
        input("waiting...")

def plot_degree_distributions(chamber="senate"):
    fig, ax = plt.subplots()
    colors = ['r', 'b', 'c', 'k', 'g']
    style = [':', '-']

    for congress in SUPPORTED_CONGRESSES:
        adj_matrix = load_adjacency_matrices(congress, chamber)
        degrees = np.sum(adj_matrix, axis=0)
        max_degree = max(degrees)
        degrees = degrees / max_degree
        max_degree = 1

        x_values = np.linspace(0, max_degree, len(set(degrees)))
        y_values = np.zeros(len(x_values))
        for i, x_value in enumerate(x_values):
            y_values[i] = len([x for x in degrees if x > x_value])
        y_values = y_values / max(y_values)

        ax.plot(x_values, y_values, colors[int(congress) % len(colors)] + style[int(congress) % len(style)], label=str(congress))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# plot_degree_distributions()

# house_members, senate_members = load_members()
# G = get_cosponsorship_graph_nx("093", which_chamber="senate")
# degs = G.degree()
# sorted_degs = sorted(degs, key=degs.get, reverse=True)
# # G = G.subgraph(sorted_degs[0:40])
# # print(G.degree())
#
# labels = {node: senate_members[(int("093"), node)] for node in G.nodes()}
# pos = nx.spring_layout(G)
#
# nx.draw_networkx_nodes(G,pos,
#                        nodelist=[node for node in G.nodes() if senate_members[(int("093"), node)][1] == 100],
#                        node_color='b', alpha=0.8)
# nx.draw_networkx_nodes(G,pos,
#                        nodelist=[node for node in G.nodes() if senate_members[(int("093"), node)][1] == 200],
#                        node_color='r', alpha=0.8)
# nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
# nx.draw_networkx_labels(G, pos, labels, font_size=8)
#
# plt.show()


def get_cosponsor_pie_chart(which_chamber="senate"):
    x_axis = []
    bipartisan = []
    dem_dem = []
    rep_rep = []
    house_members, senate_members = load_members()
    members = house_members

    for congress in SUPPORTED_CONGRESSES:
        G = get_cosponsorship_graph_nx(congress, which_chamber=which_chamber)

        cosponsor_count = {
            100*100: 0,  # dem-dem
            100*200: 0,  # dem-rep
            200*200: 0   # rep-rep
        }

        for node in G.nodes():
            if (int(congress), node) not in members.keys():
                continue
            own_party = members[(int(congress), node)][1]

            neighbors = G.neighbors(node)  # get the neighbors of this node
            for neighbor in neighbors:
                if (int(congress), neighbor) not in members.keys():
                    continue
                their_party = members[(int(congress), neighbor)][1]

                if their_party * own_party not in cosponsor_count.keys():
                    continue
                cosponsor_count[own_party * their_party] = cosponsor_count.get(own_party * their_party, 0) + 1

        total_cosponsors = sum(cosponsor_count.values())
        x_axis.append(int(congress))
        bipartisan.append(cosponsor_count[100 * 200] / total_cosponsors * 100)
        dem_dem.append(cosponsor_count[100 * 100] / total_cosponsors * 100)
        rep_rep.append(cosponsor_count[200 * 200] / total_cosponsors * 100)

        print("%s Bi: %02.02f Dem: %02.02f Rep: %02.02f" % (congress, bipartisan[-1], dem_dem[-1], rep_rep[-1]))

    print("Plotting...")

    x_axis = np.array(x_axis)
    bipartisan = np.array(bipartisan)
    dem_dem = np.array(dem_dem)
    rep_rep = np.array(rep_rep)

    fig, ax = plt.subplots()
    ax.plot(x_axis, bipartisan, 'k', label="% bipartisan cosponsorship")
    ax.plot(x_axis, dem_dem, 'b', label="% democrat cosponsorship")
    ax.plot(x_axis, rep_rep, 'r', label="% republican cosponsorship")

    # Shrink current axis by 20%
    box = ax.get_position()
    plt.ylim([0, 60])
    plt.xlim([90, 112])
    plt.title(which_chamber + " Cosponsorship Breakdown")
    plt.xlabel("Congressional Session")
    plt.ylabel("Percent of total cosponsorships")
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# get_cosponsor_pie_chart()
detect_communities("house")
# plot_degree_distributions("house")
