import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import igraph
import pandas as pd

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
                members[(int(congress), i)] = (row[0], year_count[int(row[1])])
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
    Rsqr = ssreg / sstot

    if not Rval:
        #Plot R^2 value
        plt.text(0.8 * maxxd + 0.2 * minxd, 0.8 * np.max(yd) + 0.2 * np.min(yd),
                 '$R^2 = %0.2f$' % Rsqr)
        plt.text(0.8 * maxxd + 0.2 * minxd, 0.8 * np.max(yd) + 0.2 * np.min(yd) - .1,
                 '$Slope = %0.2f$' % slope)
    else:
        #Return the R^2 value:
        return Rsqr


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


def detect_communities(chamber="senate"):
    chamber_members = load_members(which_chamber=chamber)

    for congress in SUPPORTED_CONGRESSES:
        print("-" * 80 + "\nCongress: " + congress)
        G = get_cosponsorship_graph(congress, which_chamber=chamber)

        coms = G.community_walktrap(weights=G.es['weight'])
        print("Optimal modularity:", coms.optimal_count)
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
                print("-" * 40)
                print("Dems:", num_dems)
                print("Reps:", num_reps)
                dems.append(num_dems)
                reps.append(num_reps)
                x_axis.append(i)
        plt.clf()
        plt.bar(x_axis, dems, color = 'b')
        plt.bar(x_axis, reps, color = 'r', bottom = dems)
        plt.title(str(congress) + " Congress (" + chamber.capitalize() + ") community breakdown")
        plt.savefig('renders/' + chamber + "_" + str(congress) + ".png")
        output_to_gexf(G, congress, chamber)
        print(len([cluster for cluster in clusters if len(cluster) <= 10]), "clusters with less than 10 people.")


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
        # print(node['label'], node.index)
        if (congress, int(node.index)) not in seniority_lookup.keys():
            print(0)
            continue
        years = seniority_lookup[(congress, int(node.index))][1]

        seniority.append(years)
        win_degree.append(sum(edge['weight'] for edge in G.es if edge.target == node.index))
        wout_degree.append(sum(edge['weight'] for edge in G.es if edge.source == node.index))
        in_degree.append(sum(1 for edge in G.es if edge.target == node.index))
        out_degree.append(sum(1 for edge in G.es if edge.source == node.index))
        degree.append(G.degree(node))


    plt.scatter(seniority, in_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("In Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_in_deg")

    plt.clf()
    plt.scatter(seniority, out_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Out Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_out_deg")

    plt.clf()
    plt.scatter(seniority, np.add(np.array(out_degree), np.array(in_degree)))
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_deg")

    plt.clf()
    plt.scatter(seniority, win_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted In Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_w_in_deg")

    plt.clf()
    plt.scatter(seniority, wout_degree)
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted Out Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_w_out_deg")

    plt.clf()
    plt.scatter(seniority, np.add(np.array(wout_degree), np.array(win_degree)))
    plt.xlabel("Years in the " + which_chamber.capitalize())
    plt.ylabel("Weighted Degree")
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\" + which_chamber + "_" + str(congress) + "_w_deg")


def plot_degree_distribution(which_chamber="senate", weighted=False):
    data = []
    for congress in SUPPORTED_CONGRESSES[0:4]:
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

    def animate_log_log(nframe):
        print("log-log", nframe)
        plt.plot(data[nframe][6], data[nframe][7])  # logK, logNK
        plt.title("Log-Log: " + data[nframe][8])
        plt.xlabel("log(K)")
        if weighted:
            plt.xlabel("log(Weighted K)")
        plt.ylabel("log(N(K))")
        plt.ylim(min(min(data, key=lambda x: min(x[7]))[7]), max(max(data, key=lambda x: max(x[7]))[7]))
        plt.xlim(min(min(data, key=lambda x: min(x[6]))[6]), max(max(data, key=lambda x: max(x[6]))[6]))
        trendline(data[nframe][6], data[nframe][7])
        # plt.savefig(u"C:\\Users\\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\degree_distr_log_log_plt_%s" % data[nframe][8])


    def animate_regular(nframe):
        print("regular", nframe)
        plt.bar(data[nframe][4], data[nframe][5])
        plt.title(data[nframe][8])
        plt.xlabel("K")
        if weighted:
            plt.xlabel("Weighted K")
        plt.ylabel("N(K)")
        plt.ylim(0, max(max(data, key=lambda x: max(x[5]))[5]))
        plt.savefig(u"C:\\Users\\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\degree_distr_plt_%s" % data[nframe][8])


    # now we've collected all the data
    fig = plt.figure(figsize=(5,4))
    anim = animation.FuncAnimation(fig, animate_log_log, frames=len(data))
    weighted_str = ""
    if weighted:
        weighted_str = " (weighted)"
    # plt.show()
    anim.save(u"C:\\Users\\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\gifs\\degree_distr_" + which_chamber + weighted_str + "_log_lot_plt.gif", writer='imagemagick', fps=1)
    # fig = plt.figure(figsize=(5,4))
    # anim = animation.FuncAnimation(fig, animate_regular, frames=len(data))
    # weighted_str = ""
    # if weighted:
    #     weighted_str = " (weighted)"
    # anim.save(u"C:\\Users\\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\gifs\\degree_distr_" + which_chamber + weighted_str + "_plt.gif", writer='imagemagick', fps=1)




def plot_graph_diameter():
    color = {"house": "k", "senate": "k:"}
    fig, ax = plt.subplots()

    for chamber in ["house", "senate"]:
        x_values = [int(x) for x in SUPPORTED_CONGRESSES]
        y_values = []
        for congress in SUPPORTED_CONGRESSES:
            G = get_cosponsorship_graph(congress, chamber, return_largest_connected_only=True)
            y_values.append(G.diameter(G.es['weight']))
        ax.plot(x_values, y_values, color[chamber], label=str(chamber), lw=2)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("C:\\Users\Tom\\Documents\\fall_2015\\p2psocnet\\congress_cosponsorship\\renders\\congress_diameter")


# get_cosponsor_pie_chart()
# detect_communities("senate")
# plot_degree_distributions("house")
# plot_seniority_degree(110, "house")
#

plot_degree_distribution("senate", weighted=False)
# plot_degree_distribution("senate", weighted=True)
# plot_degree_distribution("house", weighted=False)
# plot_degree_distribution("house", weighted=True)
# plot_graph_diameter()