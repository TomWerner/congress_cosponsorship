import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import community

myPath = os.path.dirname(os.path.realpath(__file__))

HOUSE_CSV_PATH = myPath + "/house_full.csv"
SENATE_CSV_PATH = myPath + "/senate_full.csv"
HOUSE_MEMBERS_FOLDER = myPath + "/cosponsorship2010/house_members"
SENATE_MEMBERS_FOLDER = myPath + "/cosponsorship2010/senate_members"
HOUSE_MATRIX_FOLDER = myPath + "/cosponsorship2010/house_matrices"
SENATE_MATRIX_FOLDER = myPath + "/cosponsorship2010/senate_matrices"
SUPPORTED_CONGRESSES = ["%03d" % x for x in range(93, 111)]
# SUPPORTED_CONGRESSES = ["%03d" % x for x in range(110, 111)]


# (congress #, icpsr #) -> party #
def load_party_data():
    hReader = csv.reader(open(HOUSE_CSV_PATH, 'r'))
    hReader.__next__()
    sReader = csv.reader(open(SENATE_CSV_PATH, 'r'))
    sReader.__next__()
    houseLookup = {}
    senateLookup = {}
    
    for row in hReader:
        houseLookup[(int(row[0]), int(row[1]))] = int(row[5])

    for row in sReader:
        senateLookup[(int(row[0]), int(row[1]))] = int(row[5])

    return houseLookup, senateLookup


# (congress #, row #) -> (name, party #)
def load_members():
    houseLookup, senateLookup = load_party_data()
    houseMembers = {}
    senateMembers = {}

    for congress in SUPPORTED_CONGRESSES:
        hReader = csv.reader(open(HOUSE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_house.txt", 'r'))
        sReader = csv.reader(open(SENATE_MEMBERS_FOLDER + "/" + str(int(congress)) + "_senators.txt", 'r'))

        for i, row in enumerate(hReader):
            if "NA" in row[2]:
                #if int(congress) == 93: print(i)
                continue
            if (int(congress), int(row[2])) in houseLookup.keys():
                houseMembers[(int(congress), i)] = (row[0], houseLookup[(int(congress), int(row[2]))])
    
        for i, row in enumerate(sReader):
            if "NA" in row[2]: continue
            if (int(congress), int(row[2])) in senateLookup.keys():
                senateMembers[(int(congress), i)] = (row[0], senateLookup[(int(congress), int(row[2]))])

    return houseMembers, senateMembers


def load_matrices():
    houseMembers, senateMembers = load_members()
    myPath = os.path.dirname(os.path.realpath(__file__))

    for congress in SUPPORTED_CONGRESSES:
        for members, reader, which in [[houseMembers, "house", "house"], [senateMembers, "sen", "senate"]]:
            G = nx.Graph()
            reader = csv.reader(open(myPath+"/cosponsorship2010/%s_matrices/%s_%smatrix.txt" % (which, congress, reader), 'r'))
            
            hData = []
            for row in reader: hData.append(row)
            #print(len(hData))

            for i in range(len(hData)):
                G.add_node(i)

            for billCol in range(len(hData[0])):
                cosponsors = []
                for i in range(len(hData)):
                    if int(hData[i][billCol]) != 0:
                        cosponsors.append(i)
                #print(cosponsors)
                for pair in itertools.combinations(cosponsors, 2):
                    G.add_edge(pair[0], pair[1])

            parties = {}
            communities = list(nx.k_clique_communities(G, 10))
            for com in communities:
                people = [members.get((int(congress),node),("",-1)) for node in com]
                print("Dem:",len([elem[1] for elem in people if elem[1] == 100]))
                print("Rep:",len([elem[1] for elem in people if elem[1] == 200]))
                print("Oth:",len([elem[1] for elem in people if elem[1] != 100 and elem[1] != 200]))
                for elem in people:
                    print(elem[0])
                print("\n"*10)
                
            #dendogram = community.generate_dendogram(G)
            #partition = community.partition_at_level(dendogram, (len(dendogram) - 2))

            #largest connected component
            #connected = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
            #partition = community.best_partition(connected)

            # partition -> [party]
            #parties = {}
            #for com in set(partition.values()):
            #    parties[com] = [members.get((int(congress),node),("",-1)) for node in partition.keys() if partition[node] == com]

            #print("Congress (%s):" % which, int(congress), ", ", len(set(partition.values())), "partitions")
            #for party in parties.keys():
            #    print("Dem:",len([elem[1] for elem in parties[party] if elem[1] == 100]))
            #    print("Rep:",len([elem[1] for elem in parties[party] if elem[1] == 200]))
                #print("Others:",[elem for elem in parties[party] if elem[1] != 100 and elem[1] != 200])
            #    print()
            print("-"*20)
        print("-"*40)
        
        #return G,parties, houseMembers

#load_matrices()



def plot_degree_distributions():
    house_members, senate_members = load_members()
    my_path = os.path.dirname(os.path.realpath(__file__))
    fig, ax = plt.subplots()

    for congress in SUPPORTED_CONGRESSES:
        for members, reader, which,line in [[house_members, "house", "house", ':'], [senate_members, "sen", "senate", '-']]:
            reader = csv.reader(open(my_path+"/cosponsorship2010/%s_matrices/%s_%smatrix.txt" % (which, congress, reader), 'r'))

            hData = []
            for row in reader: hData.append(row)
            G = np.zeros((len(hData), len(hData)))

            for billCol in range(len(hData[0])):
                cosponsors = []
                for i in range(len(hData)):
                    if int(hData[i][billCol]) != 0:
                        cosponsors.append(i)
                for pair in itertools.combinations(cosponsors, 2):
                    G[pair[0]][pair[1]] += 1
                    G[pair[1]][pair[0]] += 1
                if len(cosponsors) == 1:
                    G[cosponsors[0]][cosponsors[0]] += 1

            degrees = np.sum(G, axis=0)
            max_degree = max(degrees)
            degrees = degrees / max_degree
            max_degree = 1

            x_values = np.linspace(0, max_degree, len(set(degrees)))
            y_values = np.zeros(len(x_values))
            print(degrees)
            for i, x_value in enumerate(x_values):
                y_values[i] = len([x for x in degrees if x > x_value])
            y_values = y_values / max(y_values)

            ax.plot(x_values, y_values, 'k'+line, label=str(which) + str(congress))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()





plot_degree_distributions()