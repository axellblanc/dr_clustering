#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:40:22 2018

@author: axel
"""


import csv;
from sklearn import cluster;
import networkx as nx;
from collections import defaultdict;
import matplotlib.pyplot as plt;
from matplotlib import cm;
import seaborn as sns;
import pandas as pd;
import numpy as np;
from sklearn.metrics.cluster import normalized_mutual_info_score;
from sklearn.metrics.cluster import adjusted_rand_score;
from sklearn.preprocessing import normalize;
from sklearn.preprocessing import StandardScaler;
from sklearn import preprocessing;
from scipy.spatial.distance import cdist;
from sklearn.decomposition import PCA;
import matplotlib.backends.backend_pdf;
from PyPDF2 import PdfFileMerger, PdfFileWriter, PdfFileReader;
import io;
from reportlab.pdfgen import canvas;
from reportlab.lib.pagesizes import letter;
from collections import Counter;







#Parameters :

K = 8;
nb_PCA_components = 14
nb_drivers = 1000; #nb of travels for clustering
nb_drivers_choices = 1000; #nb of travels for choosing the PCA components and K numbers
k_range = 20; #The range in which we will search the best K
address = '/Users/axel/Desktop/driver_classification/';



DATA_OBSERVATION = False;
PCA_CHOICE = False;
K_CHOICE = False;
CLUSTER_COMPOSITION = False;
PCA_RESULTS = False;
INCLUDING_DATA_PLOT = False; #Not really sensible, would take too long
FEATURES_RESULTS = True;
INCLUDING_DATA_PLOT_2 = False;#Same as just above
VARIANCES_PER_FEATURES = True;
GROUP_BY_ID = True;
CORRELATION_WITH_CRASHES = True;
INCLUDE_CRASHES = False;
FULL_RUN = True;





#1 : Features 


selected_Features = [
'loc_speed_cnt_100',
'loc_speed_cnt_115',
'loc_speed_dur_100',
'loc_speed_dur_115',
'handling_cnt_low',
'handling_cnt_high',
'glb_speed_cnt',
'glb_speed_dur',
'hard_braking_cnt',
'rapid_acceleration_cnt',
'crash_cnt',
'handling_cnt_driver_high',
'handling_cnt_total',
'fix_speed_cnt_65',
'fix_speed_dur_65',
'crash_cnt_low',
'crash_cnt_med',
'crash_cnt_high',
'loc_speed_cnt_120',
'loc_speed_cnt_110',
'loc_speed_dur_110',
'loc_speed_dur_120',
'glb_speed_lim',
'max_speed',
'handling_cnt_non_driver',
'start_time_local',
'end_time_local'
];
        

total_Features = [
'trip_id',
'process_time',
'loc_speed_cnt_100',
'loc_speed_cnt_110',
'loc_speed_cnt_115',
'loc_speed_cnt_120',
'loc_speed_dur_100',
'loc_speed_dur_110',
'loc_speed_dur_115',
'loc_speed_dur_120',
'handling_cnt_low',
'handling_cnt_high',
'glb_speed_cnt',
'glb_speed_dur',
'glb_speed_lim',
'hard_braking_cnt',
'rapid_acceleration_cnt',
'user_id',
'company_id',
'distance',
'duration',
'start_time',
'start_time_local',
'end_time',
'end_time_local',
'score',
'validity',
'max_speed',
'crash_cnt',
'platform',
'app_version',
'sdk_version',
'sal_version',
'handling_cnt_driver_high',
'handling_cnt_non_driver',
'handling_cnt_driver_high',
'handling_cnt_total',
'email',
'mode_of_transportation',
'fix_speed_cnt_65',
'fix_speed_dur_65',
'faster_car_mode',
'os_version',
'device',
'model',
'geo_lat_last',
'geo_long_last',
'event_ts_local_last',
'event_ts_last',
'crash_cnt_low',
'crash_cnt_med',
'crash_cnt_high',];

        
distance_features = [#Features that will be divided by the distance of the trip
'loc_speed_cnt_100',
'loc_speed_cnt_110',
'loc_speed_cnt_115',
'loc_speed_cnt_120',
'loc_speed_dur_100',
'loc_speed_dur_110',
'loc_speed_dur_115',
'loc_speed_dur_120',
'glb_speed_cnt',
'hard_braking_cnt',
'rapid_acceleration_cnt',
'crash_cnt',
'fix_speed_cnt_65',
'crash_cnt_low',
'crash_cnt_med',
'crash_cnt_high',
        ];
        
time_features = [#Features that will be divided by the duration of the trip
'handling_cnt_low',
'handling_cnt_high',
'glb_speed_dur',
'handling_cnt_driver_high',
'handling_cnt_non_driver',
'handling_cnt_total',
'fix_speed_dur_65',        
        ]
        
        


#2: Data Extraction
def group_by_id(data):
    a = Counter(data[:,0]);
    idx = list(a.elements());
    idx = list(set(idx));
    id = [[i] for i in idx];
    print('There are ', len(id), ' drivers');
    for i in range(len(data)):
        for j in range(len(id)):
            if(len(id[j]) > 1):
                if(data[i,0] == id[j][0]):
                    id[j][1:] += data[i,1:].copy();
                    break;
            if(len(id[j]) == 1):
                if(data[i,0] == id[j]):
                    id[j] = data[i].copy();
                    break;
    for l in range(len(id)):
        id[l] = id[l]/a[id[l][0]];
    return np.array(id);
                    
                    
            
            

def getdata(file_name, selected_features, nb_drivers):
    data = [];
    with open(file_name) as csv_file:
        csv_reader = csv.DictReader(csv_file);
        line_count = 0;
        for row in csv_reader:
            if(row['mode_of_transportation'] == 'car' and
               float(row['loc_speed_cnt_100']) >= float(row['loc_speed_cnt_110']) and float(row['loc_speed_cnt_110'])>= float(row['loc_speed_cnt_115']) and float(row['loc_speed_cnt_115'])>= float(row['loc_speed_cnt_120']) and
               float(row['loc_speed_cnt_120']) < 10 and float(row['loc_speed_cnt_100']) < 40 and
               float(row['distance']) < 250000 and
               float(row['max_speed']) < 120):
                line = [];
                if(GROUP_BY_ID):
                    line.append(float(row['user_id']));
                distance = float(row['distance']);
                for feature in selected_features:
                    if (feature in distance_features):
                        line.append(float(row[feature])/float(row['distance']));
                    elif(feature in time_features):
                        line.append(float(row[feature])/float(row['duration']));
                    elif(feature == 'glb_speed_lim'):
                        line.append(float(row["max_speed"])/float(row[feature]));
                    elif(feature == 'start_time_local' or feature == 'end_time_local'):
                        line.append(float(row[feature][11:13]));
                    else:
                        line.append(float(row[feature]));
                if(not INCLUDE_CRASHES):
                    line.append(float(row['crash_cnt']));
                    line.append(float(row['crash_cnt_low']));
                    line.append(float(row['crash_cnt_med']));
                    line.append(float(row['crash_cnt_high']));
                data.append(line);
                line_count += 1;
            if(line_count == nb_drivers):
                break;a
        print(f'Processed {line_count} lines.')
    data = np.array(data);
    if(GROUP_BY_ID):
            data2 = group_by_id(data);
    if(not INCLUDE_CRASHES):
        crashes = data2[:,-4:].copy();
        np.delete(data2, 0, 1);
        np.delete(data2, -1, 1); 
        np.delete(data2, -1, 1); 
        np.delete(data2, -1, 1);
        np.delete(data2, -1, 1);
    else:crashes = ['no need for crashes if we take them into account in the clustering'];
    return np.array(StandardScaler().fit_transform(data2)), crashes;  
 



#3 : Get the data

if(DATA_OBSERVATION or
CLUSTER_COMPOSITION or
PCA_RESULTS or
FEATURES_RESULTS or 
VARIANCES_PER_FEATURES or
CORRELATION_WITH_CRASHES or
INCLUDE_CRASHES or
FULL_RUN ):
    data, crashes = getdata(address +'report_trip_201808211027.csv', selected_Features, nb_drivers);
    
if(PCA_CHOICE or
K_CHOICE ):
    data_choices, crashes_choices = getdata(address + 'report_trip_201808211027.csv', selected_Features, nb_drivers_choices);




# Data observation
if(FULL_RUN):
    DATA_OBSERVATION = True;
    PCA_CHOICE = True;
    K_CHOICE = True;
    CLUSTER_COMPOSITION = True;
    PCA_RESULTS = True;
    INCLUDING_DATA_PLOT = True; 
    FEATURES_RESULTS = True;
    INCLUDING_DATA_PLOT_2 = True;
    VARIANCES_PER_FEATURES = True;
    CORRELATION_WITH_CRASHES = True;


if(DATA_OBSERVATION):
    pdf_data_obs = matplotlib.backends.backend_pdf.PdfPages(address + "results/data_observation.pdf"); 
    for i in range(len(selected_Features)):
        fig = plt.figure();
        x = [];
        y = [];
        for k in range(len(data)):
            x.append(data[k][i]);
            y.append(data[k][i]);
        plt.scatter(x,y);
        plt.xlabel(selected_Features[i]);
        plt.ylabel(selected_Features[i]);
        plt.title('Observation of the selected features, with' + str(nb_drivers) + " events");
        pdf_data_obs.savefig(fig);
        plt.show();
    pdf_data_obs.close();
    
    

#4 : Chose the P.C.A.

if(PCA_CHOICE):
    pdf_pca_choice = matplotlib.backends.backend_pdf.PdfPages(address + "results/PCA_choice.pdf"); 
    pca_variances_ratio = [];
    pca_range = range(0,len(selected_Features));
    for i in pca_range:
        pca = PCA(n_components=i);
        pca.fit(data_choices);
        pca_variances_ratio.append(sum(pca.explained_variance_ratio_));
    fig = plt.figure();
    plt.plot(pca_range, pca_variances_ratio, 'bx-');
    plt.xlabel('n_components');
    plt.ylabel('Percentage of Variance');
    plt.title('Choosing an adequate pca components number, on '+ str(nb_drivers_choices) + " events");
    plt.plot(pca_range, [0.85 for i in pca_range],label = '85%', color = 'r');
    plt.plot(pca_range, [0.90 for i in pca_range],label = '90%', color = 'aquamarine');
    plt.plot(pca_range, [0.95 for i in pca_range],label = '95%', color = 'springgreen');
    plt.plot(pca_range, [0.99 for i in pca_range],label = '99%', color = 'lawngreen');
    plt.legend();
    plt.show();
    pdf_pca_choice.savefig(fig);
    pdf_pca_choice.close();




#5 : Apply Principal Component Analysis

if(K_CHOICE or CLUSTER_COMPOSITION or PCA_RESULTS or INCLUDING_DATA_PLOT or FEATURES_RESULTS or INCLUDING_DATA_PLOT_2 or VARIANCES_PER_FEATURES or CORRELATION_WITH_CRASHES):
    pca = PCA(n_components=nb_PCA_components);
    pca.fit(data);
    print('Sum of the Variances percentages :',sum(pca.explained_variance_ratio_) );
    data_pca = pca.transform(data);





#6 : Choose K 

if(K_CHOICE):
    pdf_K_choice = matplotlib.backends.backend_pdf.PdfPages(address + "results/K_choice.pdf"); 
    pca_choices = PCA(n_components=nb_PCA_components);
    pca_choices.fit(data_choices);
    data_pca_choices = pca_choices.transform(data_choices);
    distortions = [];
    K_range = range(2,k_range);
    for k in K_range:
        kmeanModel = cluster.KMeans(n_clusters=k).fit(data_pca_choices);
        kmeanModel.fit(data_pca_choices);
        distortions.append(sum(np.min(cdist(data_pca_choices, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data_choices.shape[0]);
    fig = plt.figure();
    plt.plot(K_range, distortions, 'bx-');
    plt.xlabel('k');
    plt.ylabel('Distortion');
    plt.title('The Elbow Method showing the optimal k, on ' + str(nb_drivers_choices) + " events");
    plt.show();
    pdf_K_choice.savefig(fig);
    pdf_K_choice.close();






#7 : Apply Kmeans clustering to the data


if(CLUSTER_COMPOSITION or PCA_RESULTS or INCLUDING_DATA_PLOT or FEATURES_RESULTS or INCLUDING_DATA_PLOT_2 or VARIANCES_PER_FEATURES or CORRELATION_WITH_CRASHES):
    results = [];
    algorithms = {};

    algorithms['kmeans'] = cluster.KMeans(n_clusters=K);

    '''
    #algorithms['agglom'] = cluster.AgglomerativeClustering(n_clusters=K, linkage="ward");
    #algorithms['spectral'] = cluster.SpectralClustering(n_clusters=K, affinity="precomputed");
    #algorithms['affinity'] = cluster.AffinityPropagation(damping=0.6);
    '''
    
    for model in algorithms.values():
        model.fit(data_pca);
        results.append(list(model.labels_));
    y_kmeans = algorithms['kmeans'].predict(data_pca);
    centers = algorithms['kmeans'].cluster_centers_;






# PLOT THE RESULTS:

# The composition of the clusters:

colors = ['mediumorchid','mediumpurple','steelblue','skyblue','lightseagreen','lightgreen','greenyellow','gold'];

if(CLUSTER_COMPOSITION):
    pdf_cluster_composition = matplotlib.backends.backend_pdf.PdfPages(address + "results/Dataset_repartition.pdf"); 
    occ = np.zeros(K);
    for i in y_kmeans:
        occ[i] +=1;
    fig = plt.figure();
    plt.bar(range(K), occ, color = colors);
    plt.xlabel('clusters' );
    plt.ylabel('Population');
    plt.title('Repartition of the dataset');
    pdf_cluster_composition.savefig(fig);
    plt.plot();
    pdf_cluster_composition.close();





#The PCA results

if(PCA_RESULTS):
    pdf_pca_results = matplotlib.backends.backend_pdf.PdfPages(address + "results/PCA_results.pdf"); 
    fig = plt.figure();
    pdf_pca_results.savefig(fig);
    for i in range(len(pca.components_)):
        for j in range(i,len(pca.components_)):
            if(INCLUDING_DATA_PLOT):
                fig = plt.figure();
                plt.scatter(data_pca[:,i], data_pca[:,j], c=y_kmeans, s=50, cmap='viridis');
                plt.xlabel('PCA component nb ' + str(i));
                plt.ylabel('PCA component nb ' + str(j));
                plt.title('Clusters distributions');
                pdf_pca_results.savefig(fig);
            fig = plt.figure();
            plt.scatter(centers[:, i], centers[:, j], c=np.arange(K), s=50, alpha=0.5);
            plt.xlabel('PCA component nb ' + str(i));
            plt.ylabel('PCA component nb ' + str(j));
            plt.title('Centers of the clusters');
            pdf_pca_results.savefig(fig);
    for j in range(len(pca.components_)):  
        fig = plt.figure();
        plt.title('PCA component nb ' + str(j));
        plt.xticks(rotation=90);
        ax = fig.add_subplot(111);
        x_coordinates = np.arange(len(pca.components_[j]));
        ax.bar(x_coordinates, pca.components_[j], align='center');
        ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates));
        ax.xaxis.set_major_formatter(plt.FixedFormatter(selected_Features));
        pdf_pca_results.savefig(fig);
    #We write the results in a pdf:
    pdf_pca_results.attach_note("number of PCA components: "+ str(nb_PCA_components) + "\n number of clusters K:" + str(K) + "\n We then retain "+ str(100*sum(pca.explained_variance_ratio_)) + "% of the variance"); 
    d = pdf_pca_results.infodict();
    d['Title'] = address + "results/PCA_results.pdf";
    d['Author'] = u'Axel'
    d['Subject'] = 'Drivers clustering'
    d['Keywords'] = 'Drivers clustering'
    pdf_pca_results.close();
    packet = io.BytesIO();
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=letter);
    can.drawString(10, 200, "With :");
    can.drawString(10, 190, "PCA: "+str(nb_PCA_components)+"K :" + str(K)  );
    can.drawString(10, 180, "We obtained the following results:" );
    can.drawString(10, 170,  "Variance retained with the P.C.A. : " + str(sum(pca.explained_variance_ratio_)*100) + "%");
    can.save();
    #move to the beginning of the StringIO buffer
    packet.seek(0);
    new_pdf = PdfFileReader(packet);
    # read your existing PDF
    s = address + "results/PCA_results.pdf";
    existing_pdf = PdfFileReader(open(s, "rb"));
    output = PdfFileWriter();
    # add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.getPage(0);
    page.mergePage(new_pdf.getPage(0));
    output.addPage(page);
    # finally, write "output" to a real file
    outputStream = open(address + "results/Infos.pdf", "wb");
    output.write(outputStream);
    outputStream.close();
    # merge the created pdfs
    pdfs = [address + "results/Infos.pdf", address + "results/PCA_results.pdf"];
    merger = PdfFileMerger();
    for pdf in pdfs:
        merger.append(open(pdf, 'rb'));
    with open(address + 'results/PCA_results.pdf', 'wb') as fout:
        merger.write(fout);
            


    


#The original features results



if(FEATURES_RESULTS or VARIANCES_PER_FEATURES or CORRELATION_WITH_CRASHES):
    data_reverse = pca.inverse_transform(data_pca);
    for model in algorithms.values():
        model.fit(data_reverse);
        results.append(list(model.labels_));
    y_kmeans_reverse = algorithms['kmeans'].predict(data_reverse);
    centers_reverse = algorithms['kmeans'].cluster_centers_;

if(CORRELATION_WITH_CRASHES):
    crashes_names = ['crash_cnt','crash_cnt_low',
'crash_cnt_med',
'crash_cnt_high'];

    pdf_crashes_results = matplotlib.backends.backend_pdf.PdfPages(address + "results/correlation_with_crashes.pdf"); 
    l = [[0 for i in range(K)] for j in range(4)];
    for i in range(len(y_kmeans_reverse)):
        for j in range(4):
            l[j][y_kmeans_reverse[i]] += crashes[i][j];
    for i in range(4):
        fig = plt.figure();
        plt.xlabel('Clusters' );
        plt.ylabel('Crashes');
        plt.title('Correlation between the clusters and the '+ crashes_names[i]);
        plt.bar(range(K), l[i], color = colors);
        pdf_crashes_results.savefig(fig);
    pdf_crashes_results.close()

if(FEATURES_RESULTS):
    pdf_features_results = matplotlib.backends.backend_pdf.PdfPages(address + "results/Features_results.pdf"); 
    for i in range(len(selected_Features)):
        for j in range(i,len(selected_Features)):
            if(INCLUDING_DATA_PLOT_2):
                fig = plt.figure();
                plt.scatter(data_reverse[:,i], data_reverse[:,j], c=y_kmeans_reverse, s=50, cmap='viridis');
                plt.xlabel(selected_Features[i]);
                plt.ylabel(selected_Features[j]);
                plt.title('Clusters distributions');
                pdf_features_results.savefig(fig);
            fig = plt.figure();
            plt.scatter(centers_reverse[:, i], centers_reverse[:, j], c=np.arange(K), s=50, alpha=0.5);
            plt.xlabel(selected_Features[i]);
            plt.ylabel(selected_Features[j]);
            plt.title('Centers of the clusters');
            pdf_features_results.savefig(fig);
    pdf_features_results.close();

    
if(VARIANCES_PER_FEATURES):        
    pdf2 = matplotlib.backends.backend_pdf.PdfPages(address + "results/Cluster_variances_per_features.pdf");     
    for i in range(len(selected_Features)):
        centers_x = [];
        for k in range(len(centers_reverse)):
            centers_x.append(centers_reverse[k,i]);
            m = min(centers_x);
            M = max(centers_x);
        for j in range(len(centers_x)):
            centers_x[j] = (centers_x[j]-m)/(M-m);            
        fig = plt.figure();
        plt.bar(range(len(centers_x)), centers_x, color = colors);
        plt.title(selected_Features[i]);
        pdf2.savefig(fig);  
        plt.show(); 
    pdf2.close();
     





