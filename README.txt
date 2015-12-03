Congressional Cosponsorship Network Analysis

===================
Prereqs for running
 - Install hdf5 on your system (http://www.hdfgroup.org/ftp//HDF5/current/src/unpacked/release_docs/INSTALL)
   - Windows is not recommended
 - Install Gephi - beware of Java 8, it doesn't work with Gephi (http://gephi.github.io/users/install)
 - We recommend using anaconda 3.4 - other versions have not been tested
 - pip install -r requirements.txt


===================
Running the code
 - python project.py
 - You should see it load data from pickles and the hdf5 matrices
 - It will then regenerate all the graphs found in /renders, except the gephi renders
 - It will regenerate all .gexf files in /graph_files
 - It will print out various things, most notably:
   - after number 9 is printed, you should see the
     - HITS Senate Authorities, then Hubs
     - HITS House Authorities, then Hubs
 - The total process will take about 5 minutes
Making pretty pictures
 - Open gephi, and then open the desired .gexf file from /graph_files
 - Run the weighted degree statistics on the right
 - Choose Fruchterman-Reingold from the panel on the left, and let it run until it stops moving significantly
 - In the top left panel, choose weighted in degree, and then select the ruby icon that has the hover text for "node size"
 - You can now use output the graph as a .png
   - The 93rd, 100th, and 110th House and Senate have been rendered and are in /renders
   - gephi_house_##.png and gephi_senate_##.png