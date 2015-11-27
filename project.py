import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
import h5py
import numpy as np
import igraph
import time

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

# (chamber, congress) -> adj_matrix
adj_matrix_lookup = {}

# (chamber, congress) -> nx graph
nx_graph_lookup = {}

# (chamber, congress) -> igraph graph
igraph_graph_lookup = {}

# (chamber) -> member dict
member_lookup = {}

# (chamber) - > party dict
party_lookup = {}


# (congress #, icpsr #) -> party #
def load_party_data(which_party="senate"):
    if which_party in party_lookup.keys():
        return party_lookup[which_party]

    if which_party == "senate":
        reader = csv.reader(open(SENATE_CSV_PATH, 'r'))
    else:
        reader = csv.reader(open(HOUSE_CSV_PATH, 'r'))
    reader.__next__()
    party_dict = {}

    for row in reader:
        party_dict[(int(row[0]), int(row[1]))] = int(row[5])

    party_lookup[which_party] = party_dict

    return party_dict


# (congress #, row #) -> (name, party #)
def load_members(which_chamber="senate"):
    if which_chamber in member_lookup.keys():
        return member_lookup[which_chamber]
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
    member_lookup[which_chamber] = members
    return members


# (congress #, row #) -> (name, years_in_chamber)
def load_member_seniority(which_chamber="senate"):
    # Thomas ID -> years_in_chamber
    year_count = {}
    members = {}

    for congress in SUPPORTED_CONGRESSES:
        if which_chamber == "senate":
            reader = csv.reader(open(SENATE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_senators.txt", 'r'))
        else:
            reader = csv.reader(open(HOUSE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_house.txt", 'r'))

        for i, row in enumerate(reader):
            year_count[int(row[1])] = year_count.get(int(row[1]), 0) + 2  # 2 years per congress
            if "NA" in row[2]:
                continue
            if int(row[1]) in year_count.keys():
                members[(congress, i)] = (row[0], year_count[int(row[1])])
    return members


def trendline(xd, yd, order=1, c='r', alpha=1, Rval=False):
    """Make a line of best fit"""

    #Calculate trendline
    coeffs = np.polyfit(np.array(xd), np.array(yd), order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    if order == 2: power = coeffs[0]
    else: power = 0

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept

    #Plot trendline
    plt.plot(xl, yl, c, alpha=alpha)

    #Calculate R Squared
    p = np.poly1d(coeffs)

    ybar = np.sum(yd) / len(yd)
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr = float(ssreg / sstot)

    if not Rval:
        #Plot R^2 value
        plt.text(0.8 * maxxd + 0.2 * minxd, 0.8 * np.max(yd) + 0.2 * np.min(yd),
                 '$R^2 = %0.2f$' % Rsqr)
        plt.text(0.8 * maxxd + 0.2 * minxd, 0.8 * np.max(yd) + 0.2 * np.min(yd) - .1,
                 '$Slope = %0.2f$' % slope)
    # else:
    #     #Return the R^2 value:
    #     return Rsqr
    return slope, Rsqr


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


def load_adjacency_matrices(which_congress, which_chamber="senate"):
    if (which_chamber, which_congress) in adj_matrix_lookup.keys():
        return adj_matrix_lookup[(which_chamber, which_congress)]
    which_congress = "%03d" % int(which_congress)

    if which_chamber == "senate":
        reader = csv.reader(open(my_path + "/cosponsorship2010/senate_matrices/%s_senmatrix.txt" % which_congress, 'r'))
    else:
        reader = csv.reader(open(my_path + "/cosponsorship2010/house_matrices/%s_housematrix.txt" % which_congress, 'r'))

    bill_data = []
    for row in reader: bill_data.append(row)
    adj_matrix = _load_adjacency_matrix(bill_data)

    adj_matrix_lookup[(which_chamber, which_congress)] = adj_matrix
    return adj_matrix


def get_cosponsorship_graph(congress, which_chamber="senate", return_largest_connected_only=True):
    if (which_chamber, congress, return_largest_connected_only) in igraph_graph_lookup.keys():
        return igraph_graph_lookup[(which_chamber, congress, return_largest_connected_only)]
    if (which_chamber, congress, False) in igraph_graph_lookup.keys() and return_largest_connected_only is True:
        G = igraph_graph_lookup[(which_chamber, congress, False)]
        components = G.components()
        # find the largest connected component
        largest_connected_component = max(components, key=lambda x: len(x))
        G = components.subgraph(list(components).index(largest_connected_component))
        igraph_graph_lookup[(which_chamber, congress, return_largest_connected_only)] = G
        return G
    print("Creating igraph graph for ", congress, which_chamber, return_largest_connected_only)

    adj_matrix = load_adjacency_matrices(congress, which_chamber)
    member_lookup = load_members(which_chamber=which_chamber)

    G = igraph.Graph()
    for i in range(len(adj_matrix)):
        if (int(congress), i) in member_lookup.keys():
            member = member_lookup[(int(congress), i)]
            G.add_vertex(i, label=member[0], party=member[1])
        else:
            G.add_vertex(i)

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

    igraph_graph_lookup[(which_chamber, congress, return_largest_connected_only)] = G
    return G


def get_cosponsorship_graph_nx(congress, which_chamber="senate", return_largest_connected_only=True):
    if (which_chamber, congress, return_largest_connected_only) in nx_graph_lookup.keys():
        return nx_graph_lookup[(which_chamber, congress, return_largest_connected_only)]
    if (which_chamber, congress, False) in nx_graph_lookup.keys() and return_largest_connected_only is True:
        G = sorted(nx.connected_component_subgraphs(igraph_graph_lookup[(which_chamber, congress, False)]), key=len, reverse=True)[0]
        nx_graph_lookup[(which_chamber, congress, return_largest_connected_only)] = G
        return G
    print("Creating nx graph for ", congress, which_chamber, return_largest_connected_only)

    adj_matrix = load_adjacency_matrices(congress, which_chamber)

    G = nx.Graph()
    for i in range(0, len(adj_matrix)): G.add_node(i)  # setup our graph

    for row in range(len(adj_matrix)):
        for col in range(row, len(adj_matrix)):
            if adj_matrix[row][col] != 0:
                G.add_edge(row, col, weight=adj_matrix[row][col])

    if return_largest_connected_only:
        G = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]

    nx_graph_lookup[(which_chamber, congress, return_largest_connected_only)] = G
    return G


def output_to_gexf(G, congress, which_house):
    file = open("graph_files/" + congress + "_" + which_house + "_cosponsorship.gexf", 'w')
    file.write("""<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2" xmlns:viz="http://www.gexf.net/1.2draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd">
    <meta lastmodifieddate="2009-03-20">
        <creator>Tom Werner</creator>
        <description>%s %s cosponsorship network</description>
    </meta>
    <graph mode="static" defaultedgetype="directed">\n""" % (congress, which_house))
    file.write("\t\t<nodes>\n")
    for node in G.vs:
        file.write('\t\t\t<node id="%d" label="%s">\n' % (node.index, node['label']))
        if node['community_number'] is not None:
            file.write('\t\t\t\t<attvalues>\n')
            file.write('\t\t\t\t\t<attvalue for="modularity_class" value="%d"></attvalue>\n' % node['community_number'])
            file.write('\t\t\t\t</attvalues>\n')
            file.write('\t\t\t\t<viz:color r="%d" g="%d" b="%d"></viz:color>\n' % (node['red'], node['green'], node['blue']))
            # if node['party'] == 100:
            #     file.write('\t\t\t\t<viz:color r="%d" g="%d" b="%d"></viz:color>\n' % (255, 0, 0))
            # if node['party'] == 200:
            #     file.write('\t\t\t\t<viz:color r="%d" g="%d" b="%d"></viz:color>\n' % (0, 0, 255))
        file.write('\t\t\t</node>\n')
    file.write("\t\t</nodes>\n")

    file.write("\t\t<edges>\n")
    for edge in G.es:
        file.write('<edge id="%d" source="%d" target="%d" weighted="%d" />\n' % (edge.index, edge.source, edge.target, edge['weight']))
    file.write("\t\t</edges>\n")
    file.write("""
    </graph>
</gexf>\n""")
    file.close()


def detect_communities(chamber="senate", print_output=False):
    chamber_members = load_members(which_chamber=chamber)

    for congress in SUPPORTED_CONGRESSES:
        result = congress + "\t"
        if print_output:
            print("-" * 80 + "\nCongress: " + congress)
        G = get_cosponsorship_graph(congress, which_chamber=chamber)

        coms = G.community_walktrap(weights=G.es['weight'])
        if print_output:
            print("Optimal modularity:", coms.optimal_count)
        result += str(coms.optimal_count) + "\t"
        clusters = coms.as_clustering(coms.optimal_count)

        x_axis = []
        dems = []
        reps = []
        for i, cluster in enumerate(clusters):
            members = [chamber_members[(int(congress), x)] for x in cluster if
                       (int(congress), x) in chamber_members.keys()]
            num_dems = len([temp for temp in members if temp[1] == 100])
            num_reps = len([temp for temp in members if temp[1] == 200])

            for node in cluster:
                if num_dems == 0 and num_reps == 0:
                    continue
                G.vs[node]['community_number'] = i
                G.vs[node]['red'] = int((num_reps / (num_dems + num_reps)) * 255)
                G.vs[node]['green'] = 0
                G.vs[node]['blue'] = int((num_dems / (num_dems + num_reps)) * 255)
            if len(cluster) > 10:
                if print_output:
                    print("-" * 40)
                    print("Dems:", num_dems)
                    print("Reps:", num_reps)
                result += str(num_dems) + "\t" + str(num_reps) + "\t"
                dems.append(num_dems)
                reps.append(num_reps)
                x_axis.append(i)
        print(result)
        plt.clf()
        plt.bar(x_axis, dems, color='b')
        plt.bar(x_axis, reps, color='r', bottom=dems)
        plt.title(str(congress) + " Congress (" + chamber.capitalize() + ") community breakdown")
        plt.savefig('renders/com_breakdown_' + chamber + "_" + str(congress) + ".png")
        output_to_gexf(G, congress, chamber)
        if print_output:
            print(len([cluster for cluster in clusters if len(cluster) <= 10]), "clusters with less than 10 people.")


def get_cosponsor_pie_chart(which_chamber="senate"):
    x_axis = []
    bipartisan = []
    dem_dem = []
    rep_rep = []
    members = load_members(which_chamber)

    for congress in SUPPORTED_CONGRESSES:
        G = get_cosponsorship_graph(congress, which_chamber=which_chamber)

        cosponsor_count = {
            100*100: 0,  # dem-dem
            100*200: 0,  # dem-rep
            200*200: 0   # rep-rep
        }

        for edge in G.es:
            source = edge.source
            target = edge.target

            if (int(congress), source) not in members.keys() or (int(congress), target) not in members.keys():
                continue
            s_party = members[(int(congress), source)][1]
            t_party = members[(int(congress), target)][1]
            if s_party * t_party not in cosponsor_count.keys():
                    continue
            cosponsor_count[s_party * t_party] = cosponsor_count.get(s_party * t_party, 0) + edge['weight']

        total_cosponsors = sum(cosponsor_count.values())
        x_axis.append(int(congress))
        bipartisan.append(cosponsor_count[100 * 200] / total_cosponsors * 100)
        dem_dem.append(cosponsor_count[100 * 100] / total_cosponsors * 100)
        rep_rep.append(cosponsor_count[200 * 200] / total_cosponsors * 100)

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
    # plt.show()
    plt.savefig(my_path + "/renders/cosponsorship_breakdown")


def plot_seniority_degree(congress, which_chamber="senate"):
    G = get_cosponsorship_graph(congress, which_chamber, False)
    seniority_lookup = load_member_seniority(which_chamber)

    seniority = []
    in_degree = []
    out_degree = []
    win_degree = []
    wout_degree = []
    degree = []

    for node in sorted([node for node in G.vs if node['label'] is not None], key=lambda x: x['label']):
        if (congress, int(node.index)) not in seniority_lookup.keys():
            print("skipping")
            continue
        years = seniority_lookup[(congress, int(node.index))][1]

        seniority.append(years)
        win_degree.append(sum(edge['weight'] for edge in G.es if edge.target == node.index))
        wout_degree.append(sum(edge['weight'] for edge in G.es if edge.source == node.index))
        in_degree.append(sum(1 for edge in G.es if edge.target == node.index))
        out_degree.append(sum(1 for edge in G.es if edge.source == node.index))
        degree.append(G.degree(node))

    print(len(G.vs))
    plt.scatter(seniority, in_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("In Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_in_deg")

    plt.clf()
    plt.scatter(seniority, out_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Out Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_out_deg")

    plt.clf()
    plt.scatter(seniority, np.add(np.array(out_degree), np.array(in_degree)))
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_deg")

    plt.clf()
    plt.scatter(seniority, win_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted In Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_w_in_deg")

    plt.clf()
    plt.scatter(seniority, wout_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted Out Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_w_out_deg")

    plt.clf()
    plt.scatter(seniority, np.add(np.array(wout_degree), np.array(win_degree)))
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted Degree")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "_w_deg")


def plot_degree_distribution(which_chamber="senate", weighted=False):
    data = []
    for congress in SUPPORTED_CONGRESSES:
        G = get_cosponsorship_graph(congress, which_chamber, return_largest_connected_only=False)
        if not weighted:
            degrees = G.degree()
            filename = which_chamber + str(congress)
        else:
            filename = which_chamber + "_weighted_" + str(congress)
            win_degree = []
            wout_degree = []
            for node in G.vs:
                win_degree.append(sum(edge['weight'] for edge in G.es if edge.target == node.index))
                wout_degree.append(sum(edge['weight'] for edge in G.es if edge.source == node.index))
            degrees = np.array(np.add(np.array(win_degree), np.array(wout_degree)))

        degree_number = np.arange(0, max(degrees) + 1)  # k
        degree_count = np.zeros(len(degree_number))  # N(k)

        for node, degree in enumerate(degrees):
            degree_count[(degree // 10) * 10] += 1

        k = []
        nK = []
        for i in range(len(degree_number)):
            if degree_count[i] != 0 and degree_number[i] != 0:
                k.append((degree_number[i] // 10) * 10)
                nK.append(degree_count[i])

        logNK = np.log(nK)
        logK = np.log(k)

        data.append((G, degrees, degree_number, degree_count, k, nK, logK, logNK, filename))

    def animate_log_log(nframe, loglog_xmin, loglog_xmax, loglog_ymin, loglog_ymax):
        plt.clf()
        plt.plot(data[nframe][6], data[nframe][7])  # logK, logNK
        plt.title("Log-Log: " + data[nframe][8])
        plt.xlabel("log(K)")
        if weighted:
            plt.xlabel("log(Weighted K)")
        plt.ylabel("log(N(K))")
        plt.xlim(loglog_xmin, loglog_xmax)
        plt.ylim(loglog_ymin, loglog_ymax)
        slope, rsqrd = trendline(data[nframe][6], data[nframe][7])
        plt.savefig(my_path + "/renders/degree_distr_log_log_plt_%s" % data[nframe][8])
        return slope, rsqrd


    def animate_regular(nframe, ymax):
        plt.clf()
        plt.bar(data[nframe][4], data[nframe][5])
        plt.title(data[nframe][8])
        plt.xlabel("K")
        if weighted:
            plt.xlabel("Weighted K")
        plt.ylabel("N(K)")
        plt.ylim(0, ymax)
        plt.savefig(my_path + "/renders/degree_distr_plt_%s" % data[nframe][8])


    # now we've collected all the data
    loglog_xmin = min(min(data, key=lambda x: min(x[6]))[6])
    loglog_xmax = max(max(data, key=lambda x: max(x[6]))[6])
    loglog_ymin = min(min(data, key=lambda x: min(x[7]))[7])
    loglog_ymax = max(max(data, key=lambda x: max(x[7]))[7])
    regular_ymax = max(max(data, key=lambda x: max(x[5]))[5])
    for i in range(len(data)):
        # print(i)
        slope, rsqrd = animate_log_log(i, loglog_xmin, loglog_xmax, loglog_ymin, loglog_ymax)
        print(which_chamber + "\t" + str(SUPPORTED_CONGRESSES[i]) + "\t" + str(weighted) + "\t" + str(slope) + "\t" + str(rsqrd))
        animate_regular(i, regular_ymax)


def plot_graph_diameter():
    color = {"house": "k", "senate": "k:"}
    fig, ax = plt.subplots()

    for chamber in ["house", "senate"]:
        x_values = [int(x) for x in SUPPORTED_CONGRESSES]
        y_values = []
        for congress in SUPPORTED_CONGRESSES:
            G = get_cosponsorship_graph(congress, chamber, return_largest_connected_only=True)
            y_values.append(G.diameter(G.es['weight']))
        ax.plot(x_values, y_values, color[chamber], label=str(chamber.capitalize()), lw=2)
        print(chamber, y_values)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.xlabel("Congress")
    plt.ylabel("Diameter")
    plt.title("Cosponsorship Network Diameter in Congress")

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(my_path + "/renders/congress_diameter")


def plot_graph_clustering_coefficient():
    color = {"house": "k", "senate": "k:"}
    fig, ax = plt.subplots()

    for chamber in ["house", "senate"]:
        x_values = [int(x) for x in SUPPORTED_CONGRESSES]
        y_values = []
        for congress in SUPPORTED_CONGRESSES:
            G = get_cosponsorship_graph(congress, chamber, return_largest_connected_only=True)
            y_values.append(G.transitivity_avglocal_undirected())
        ax.plot(x_values, y_values, color[chamber], label=str(chamber.capitalize()), lw=2)
        print(chamber, y_values)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.xlabel("Congress")
    plt.ylabel("Avg Clustering Coefficient")
    plt.title("Average Clustering Coefficient in Congress")
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(my_path + "/renders/congress_clustering_coefficient")


def load_data():
    start = time.time()
    try:
        print("Loading data from /data pickles and hfd5 adj matrices")
        f = h5py.File('data/cosponsorship_data.hdf5', 'r')
        for chamber in ['house', 'senate']:
            for congress in SUPPORTED_CONGRESSES:
                adj_matrix_lookup[(chamber, congress)] = np.asarray(f[chamber + str(congress)])

                igraph_graph = igraph.load("data/" + chamber + str(congress) + "_igraph.pickle", format="pickle")
                igraph_graph_lookup[(chamber, congress, False)] = igraph_graph

                nx_graph = nx.read_gpickle("data/" + chamber + str(congress) + "_nx.pickle")
                nx_graph_lookup[(chamber, congress, False)] = nx_graph
    except IOError as e:
        print("Loading data from cosponsorship files")
        f = h5py.File("data/cosponsorship_data.hdf5", "w")
        for chamber in ['house', 'senate']:
            for congress in SUPPORTED_CONGRESSES:
                print("Starting %s %s" % (str(congress), chamber))
                adj_matrix = load_adjacency_matrices(congress, chamber)
                data = f.create_dataset(chamber + str(congress), adj_matrix.shape, dtype='f')
                data[0: len(data)] = adj_matrix

                # igraph
                get_cosponsorship_graph(congress, chamber, False).save("data/" + chamber + str(congress) + "_igraph.pickle", "pickle")
                # networkx
                nx.write_gpickle(get_cosponsorship_graph_nx(congress, chamber, False), "data/" + chamber + str(congress) + "_nx.pickle")

                print("Done with %s %s" % (str(congress), chamber))
    print("Data loaded in %d seconds" % (time.time() - start))


def compute_hits(weighted_adj_matrix, iteration=1000):
    n = len(weighted_adj_matrix)

    Au = np.dot(weighted_adj_matrix.T, weighted_adj_matrix)
    Hu = np.dot(weighted_adj_matrix, weighted_adj_matrix.T)

    a = np.ones(n)
    h = np.ones(n)

    for j in range(iteration):
        done = False
        new_a = np.dot(a, Au)
        if np.allclose(new_a, a):
            done = True
        a = new_a
        a = a / sum(a)

        new_h = np.dot(h, Hu)
        if np.allclose(new_a, a):
            done &= True
        h = new_h
        h = h / sum(h)

        if done:
            break

    return a, h


def rank_by_hits(congress, which_chamber="senate"):
    adj_matrix = load_adjacency_matrices(congress, which_chamber)
    authority, hubs = compute_hits(adj_matrix)
    seniority = load_member_seniority(which_chamber)
    seniority_array = np.zeros(authority.shape)

    for i in range(len(authority)):
        if (congress, i) in seniority.keys():
            seniority_array[i] = seniority[(congress, i)][1]
        else:
            seniority_array[i] = -1

    plt.scatter(seniority_array, authority)
    plt.xlabel("Seniority")
    plt.ylabel("HITS Authority Score")
    plt.title(str(congress) + " " + which_chamber.capitalize() + " HITS Authority vs Seniority")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "hits_authority")
    sorted_indexes = np.argsort(authority)[::-1]
    print("-" * 80)
    for i in range(10):
        name = seniority[(congress, sorted_indexes[i])][0]
        print(" ".join(name.split('  ')[::-1]))

    plt.clf()

    plt.scatter(seniority_array, hubs)
    plt.xlabel("Seniority")
    plt.ylabel("HITS Hub Score")
    plt.title(congress.capitalize() + " HITS Hubs vs Seniority")
    plt.savefig(my_path + "/renders/" + which_chamber + "_" + str(congress) + "hits_hub")
    sorted_indexes = np.argsort(hubs)[::-1]
    print("-" * 80)
    for i in range(10):
        name = seniority[(congress, sorted_indexes[i])][0]
        print(" ".join(name.split('  ')[::-1]))




load_data()

start = time.time()

# plt.clf()
# get_cosponsor_pie_chart(which_chamber="senate")
# print(1)
# plt.clf()
# get_cosponsor_pie_chart(which_chamber="house")
# print(2)
# plt.clf()
# plot_seniority_degree('110', "senate")
# print(3)
# plt.clf()
# plot_seniority_degree('110', "house")
# print(4)
# plt.clf()
# plot_degree_distribution("senate", weighted=False)
# print(5)
# plt.clf()
# plot_degree_distribution("senate", weighted=True)
# print(6)
# plt.clf()
# plot_degree_distribution("house", weighted=False)
# print(7)
# plt.clf()
# plot_degree_distribution("house", weighted=True)
# print(8)
# plt.clf()
# plot_graph_diameter()
# print(9)
# plt.clf()
rank_by_hits("110", "senate")
print(10)
plt.clf()
# rank_by_hits("110", "house")
# print(11)
# plt.clf()
# detect_communities("senate")
# print(12)
# plt.clf()
# detect_communities("house")
# print(13)
# plt.clf()
# plot_graph_clustering_coefficient()
# print(14)
# plt.clf()
print("Done in %d seconds" % (time.time() - start))