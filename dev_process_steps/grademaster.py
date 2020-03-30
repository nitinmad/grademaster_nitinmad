#!/usr/bin/env python

SCRIPT_NAME = "grademaster"
SCRIPT_VERSION = "v0.2.2"
REVISION_DATE = "2016-11-28"
AUTHOR = """
Johannes Hachmann (hachmann@buffalo.edu) with contributions by:
   Mojtaba Haghighatlari (jobfile, Pandas dataframe) 
"""
DESCRIPTION = "This little program is designed to help manage course grades, make grade projections, etc."

# Version history timeline:
# v0.0.1 (2015-03-02): pseudocode outline 
# v0.0.2 (2015-03-02): more detailed pseudocode outline 
# v0.1.0 (2015-03-02): add basic code infrastructure from previous scripts
# v0.1.1 (2015-03-02): add basic functionality (identify data structure of input file)
# v0.1.2 (2015-03-03): add basic functionality (dictionary of dictionaries)
# v0.1.3 (2015-03-04): implement dictionary of dictionaries properly
# v0.1.4 (2015-03-04): put in some checks and read in the data into dictionary
# v0.1.5 (2015-03-04): revamp the data structure
# v0.1.6 (2015-03-04): implement grading rules
# v0.1.7 (2015-03-05): implement letter grades
# v0.1.8 (2015-03-05): fix rounding error in letter grades
# v0.1.9 (2015-03-05): some more analysis
# v0.1.10 (2015-03-05): student ranking
# v0.1.11 (2015-03-09): grades throughout the semester
# v0.1.12 (2015-03-09): cleanup and rewrite; also, clean up the print statements; introduce two other input file for debugging that are more realistic, easier; 
                         #build in a few extra safety checks  
# v0.1.13 (2015-03-09): continue cleanup and rewrite beyond data acquisition 
# v0.1.14 (2015-03-09): continue cleanup and rewrite beyond grade calculation; make letter grade conversion into function 
# v0.1.15 (2015-03-09): continue cleanup and rewrite beyond letter grade conversion; introduce custom statistics function 
# v0.1.16 (2015-03-09): continue cleanup and rewrite beyond grade statistics 
# v0.2.0  (2015-10-12): major overhaul introducing contributions from students; rename to grademaster; introduce jobfile; use of Pandas dataframes 
# v0.2.1  (2015-10-25): add requestmeeting; generalize for HW>5
# v0.2.2  (2016-11-28): generalize for >M2

###################################################################################################
# TASKS OF THIS SCRIPT:
# -assorted collection of tools for the analysis of grade data
###################################################################################################

###################################################################################################
#TODO:
# -replaced optparse with argparser
# -make use of different print levels
###################################################################################################

import sys
import os
import time
import string
import math
# import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: this should at some point replaced with argparser
# import argparse
from optparse import OptionParser
from collections import defaultdict
from operator import itemgetter

from math import sqrt
# import numpy as np

from lib_jcode import (banner,
                       print_invoked_opts,
                       tot_exec_time_str,
                       intermed_exec_timing,
                       std_datetime_str,
                       chk_rmfile
                       )
     

###################################################################################################

def percent2lettergrade(percentgrade):
    """(percent2lettergrade):
        This function converts percentgrades into lettergrades according to a given conversion scheme.
    """
# TODO: this is the place to realize different grading-schemes in a more general fashion
    if round(percentgrade) >= 96:
        return 'A'
    elif round(percentgrade) >= 91:
        return 'A-'
    elif round(percentgrade) >= 86:
        return 'B+'
    elif round(percentgrade) >= 81:
        return 'B'
    elif round(percentgrade) >= 76:
        return 'B-'
    elif round(percentgrade) >= 71:
        return 'C+'
    elif round(percentgrade) >= 66:
        return 'C'
    elif round(percentgrade) >= 61:
        return 'C-'
    elif round(percentgrade) >= 56:
        return 'D+'
    elif round(percentgrade) >= 51:
        return 'D'
    else:
        return 'F'

###################################################################################################

# TODO: we should really use a library/module for this rather than coding it by hand (this is just for exercise)
def median_func(val_list):
    """(median function):
        Takes a list of values and returns the median. 
    """
    val_list = sorted(val_list)
    # different cases
    # empty list
    if len(val_list) < 1:
        return None
    # list with odd numbers of values
    if len(val_list) %2 == 1:   
        return val_list[((len(val_list)+1)/2)-1]
    # list with even numbers of values
    if len(val_list) %2 == 0:
        return float(sum(val_list[(len(val_list)/2)-1:(len(val_list)/2)+1]))/2.0

###################################################################################################

def distribution_stat(val_list):
    """(distribution_stat):
        Takes a list and returns some distribution statistics in form of a dictionary. 
    """
    n_vals = len(val_list)
    
    if n_vals == 0:
        stat = {'n': 0, 'av': None, 'median': None, 'min': None, 'max': None, 'mad': None, 'rmsd': None, 'spread': None}
    else:
        average = 0.0
        for val in val_list:
            average += val
        average = average/n_vals 

        median = median_func(val_list)

        val_list.sort()
        min = val_list[0]
        max = val_list[-1]
        spread = abs(max - min)

        mad = 0.0
        for val in val_list:
            mad += abs(val-average)
        mad = mad/n_vals

        rmsd = 0.0
        for val in val_list:
            rmsd += (val-average)**2
        rmsd = sqrt(rmsd/n_vals)

        stat = {'n': n_vals, 'av': average, 'median': median, 'min': min, 'max': max, 'mad': mad, 'rmsd': rmsd, 'spread': spread}
    return stat

###################################################################################################

def histogram(val_list):
    """(histogram_dict):
        Takes a list and returns a dictionary with histogram data. 
    """
    dict = {}
    for x in val_list:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 1
    return dict


###################################################################################################

# TODO: this should potentially go into a library
def print_df(dataframe): 
    """(print_df):
        Prints an entire Pandas DataFrame. 
    """

    pd.set_option('display.max_rows', len(dataframe))
    print(dataframe)
    pd.reset_option('display.max_rows')
    return 

###################################################################################################

def main(opts,commline_list):
    """(main):
        Driver of the grademaster script.
    """
    time_start = time.time()

    # now the standard part of the script begins
    logfile = open(opts.logfile,'a',0)
    error_file = open(opts.error_file,'a',0)
    
    banner(logfile, SCRIPT_NAME, SCRIPT_VERSION, REVISION_DATE, AUTHOR, DESCRIPTION)

    # give out options of this run
    print_invoked_opts(logfile,opts,commline_list)

    home_dir = os.getcwd() 

    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str
    logfile.write(tmp_str + '\n')

    #################################################################################################
    # read in the CSV file with the raw data of grades 

    # make a logfile entry and screen entry so that we know where we stand
    tmp_str = "Starting data acquisition..."
    print tmp_str
    logfile.write(tmp_str + '\n')
    
    # check that file exists, get filename from optparse
    if opts.data_file is None:
        tmp_str = "... data file not specified!"
        print tmp_str
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')

        tmp_str = "Aborting due to missing data file!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)
# TODO: This should better be done by exception handling
        
    
    tmp_str = "   ...reading in data..."
    print tmp_str
    logfile.write(tmp_str + '\n')

    # open CSV file with raw data
    rawdata_df = pd.read_csv(opts.data_file)
#    print_df(rawdata_df)


    tmp_str = "   ...cleaning data structure..."
    print tmp_str
    logfile.write(tmp_str + '\n')

    # remove empty entries
    for i in rawdata_df.columns:
        if 'Unnamed'in i:
            rawdata_df = rawdata_df.drop(i,1)
    rawdata_df = rawdata_df.dropna(how='all')
#    print_df(rawdata_df)


    tmp_str = "   ...identify keys..."
    print tmp_str
    logfile.write(tmp_str + '\n')

    # read top line of data file, which defines the keys
    keys_list = list(rawdata_df.columns)   
    n_keys = len(keys_list)  
#    print keys_list


        
#     # OLD VERSION
#     # open CSV file with raw data
#     data_file = open(opts.data_file,'r')
# 
#     # read top line of data file, which defines the keys
#     line = data_file.readline()
#     # use commas for split operation
#     words = line.split(',')
#     # extract keys, get rid of empty entries
#     keys_list = []
#     for word in words:
#         if word != '' and word != '\r\n':
#             keys_list.append(word)


# TODO: we should make this more general purpose
# TODO: rewrite this in a more elegant form
    tmp_str = "   ...checking validity of data structure..."
    print tmp_str
    logfile.write(tmp_str + '\n')
    # check that the standard keys are amongst the first three keys, because that's all we have implemented so far 
    if "Last name" not in keys_list[0:4]:
        tmp_str = "   ...'Last name' missing in data structure!"
        print tmp_str
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        tmp_str = "Aborting due to invalid data structure!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)
    elif "First name" not in keys_list[0:4]:
        tmp_str = "   ...'First name' missing in data structure!"
        print tmp_str
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        tmp_str = "Aborting due to invalid data structure!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)
    elif "Student ID" not in keys_list[0:4]:
        tmp_str = "   ...'Student ID' missing in data structure!"
        print tmp_str
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        tmp_str = "Aborting due to invalid data structure!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)
    elif "email" not in keys_list[0:4]:
        tmp_str = "   ...'email' missing in data structure!"
        print tmp_str
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        tmp_str = "Aborting due to invalid data structure!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)

    
    # check if all the grades are in float type (not object)
    for i in keys_list[4:]:
        if rawdata_df[i].dtypes == object:  
            tmp_str = "Aborting due to unknown grade format in column %s!" %i 
            logfile.write(tmp_str + '\n')
            error_file.write(tmp_str + '\n')
            sys.exit(tmp_str)
     

#     # OLD VERSION
    # suitable data structure for raw and derived data: dictionary of dictionaries with mixed arguments -> stackoverflow
    # template:
    # data_dict['ID']['first_name'] = "xxx"
    # data_dict['ID']['last_name'] = "yyy"
    # data_dict['ID']['hw_grades'] = []    list of variable entries
    # data_dict['ID']['midterm_grades'] = []    list of variable entries
    # data_dict['ID']['final_grade'] = z   some number
#     data_dict = defaultdict(lambda : defaultdict(int))  # note: we use the anonymous function construct lambda here

    # make ID list since this is our distinguishing dictionary key 
#     id_list = []
#     tmp_str = "   ...reading in bulk of data..."
#     print tmp_str
#     logfile.write(tmp_str + '\n')

    # use standard read in with infinite loop construct
#     while 1:
#         line = data_file.readline()
#         if not line: break        
#         words = line.split(',')
#         # temporary data list
#         data_list = []
#         for word in words:
#             # get rid of junk data
#             if word != '' and '\r\n' not in word:   # note: we had to make junk removal more general
#                 # populate the temporary data_list
#                 data_list.append(word)  
        
#         # continue if data_list is emptycheck that we don't have an empty list
#         if len(data_list) == 0:
#             continue
#         # check that the data_list and key_list have to have the same lenght
#         elif len(data_list) != n_keys:
#             tmp_str = "   ...invalid data entry (wrong number of data entries): " + line
#             print tmp_str
#             logfile.write(tmp_str + '\n')
#             error_file.write(tmp_str + '\n')
#             tmp_str = "Aborting due to invalid data entry!"
#             logfile.write(tmp_str + '\n')
#             error_file.write(tmp_str + '\n')
#             sys.exit(tmp_str)
#         # TODO: think about a more sophisticated handling in case of problems
# 
# 
#         # find index of list element in keys_list that contains the id
#         id_index = keys_list.index("Student ID")                    
#         # get id
#         id = data_list[id_index]
#         # add id to id_list
#         id_list.append(id)
#         # set up hw and midterm lists to get added to dictionary later
#         hw_list = []
#         midterm_list = []
# 
#         for i in xrange(n_keys):    # note that we use xrange instead of range
#             key = keys_list[i]
#             data = data_list[i]
#             if key == "Last name":
#                 data_dict[id]['last_name'] = data 
#             elif key == "First name":
#                 data_dict[id]['first_name'] = data 
#             elif key == "Student ID":
#                 continue
#             elif 'HW' in key:
#                 hw_list.append(float(data))         # don't forget to convert string to float
#             elif (key == 'M1') or (key == 'M2'):    
#                 midterm_list.append(float(data))    # don't forget to convert string to float
#             elif key == 'Final':
#                 data_dict[id]['final_grade'] = float(data)  # don't forget to convert string to float
#             else:
#                 tmp_str = "Aborting due to unknown key!"
#                 logfile.write(tmp_str + '\n')
#                 error_file.write(tmp_str + '\n')
#                 sys.exit(tmp_str)
# 
#         # now we have to put lists into dictionary                    
#         data_dict[id]['hw_grades'] = hw_list
#         data_dict[id]['midterm_grades'] = midterm_list
# 
#  
#     # close file
#     data_file.close()


    # some bookkeeping on where we stand int the semester
    n_hws = 0
    n_midterms = 0
    n_final = 0
    for key in keys_list[4:]:
        if "HW" in key:
            n_hws += 1
        elif "M" in key:
            n_midterms += 1  
        elif "Final" in key:
            n_final += 1
        else:                
            tmp_str = "Aborting due to unknown key!"
            logfile.write(tmp_str + '\n')
            error_file.write(tmp_str + '\n')
            sys.exit(tmp_str)
 
 
#    print n_hws
#    print n_midterms
#    print n_final 
            
    tmp_str = "...data acquisition finished."
    print tmp_str
    logfile.write(tmp_str + '\n')
    

    #################################################################################################


    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "Summary of acquired data:"
    print tmp_str
    logfile.write(tmp_str + '\n')
    
    tmp_str = "   Number of students:  " + str(len(rawdata_df))
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of homeworks: " + str(n_hws)
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of midterms:  " + str(n_midterms)
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of finals:    " + str(n_final)
    print tmp_str
    logfile.write(tmp_str + '\n')
# TODO: this should be better formatted    

    #################################################################################################


    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "Starting calculation of grades and grade projections..."
    print tmp_str
    logfile.write(tmp_str + '\n')

#     print "Info"
#     print rawdata_df.info()
#     print "Keys"
#     print rawdata_df.keys()
#     print "Index"
#     print rawdata_df.index
#     print "Columns"
#     print rawdata_df.columns
#     print "Values"
#     print rawdata_df.values
#     print "Describe"
#     print rawdata_df.describe()


# TODO: this is very inelegant and should be changed
    # Set up projection dataframe   
    hwdata_df = rawdata_df.copy()
    examdata_df = rawdata_df.copy()
    # empty all data fields in projection_df
    for i in xrange(4,n_keys):
        key = keys_list[i]
        if 'HW' in key:
            examdata_df.drop(key, axis=1, inplace=True)
        elif key in ('M1', 'M2','F'):
            hwdata_df.drop(key, axis=1, inplace=True)

#    print hwdata_df
#    print examdata_df


    hwkeys_list = list(hwdata_df.columns)   
    n_hwkeys = len(hwkeys_list)  

    examkeys_list = list(examdata_df.columns)   
    n_examkeys = len(examkeys_list)  

    acc_hwdata_df = hwdata_df.copy()
    acc_examdata_df = examdata_df.copy()
    
    for i in xrange(4,n_hwkeys):
        key = hwkeys_list[i]
        if key == 'HW1':
            continue
        else:
            prevkey = hwkeys_list[i-1]            
            acc_hwdata_df[key] += acc_hwdata_df[prevkey]
        

    for i in xrange(4,n_examkeys):
        key = examkeys_list[i]
        if key == 'M1':
            continue
        else:
            prevkey = examkeys_list[i-1]            
            acc_examdata_df[key] += acc_examdata_df[prevkey]

#    print acc_hwdata_df
#    print acc_examdata_df

    av_hwdata_df = acc_hwdata_df.copy()
    av_examdata_df = acc_examdata_df.copy()
    minmax_midtermdata_df = examdata_df.copy()

    for i in xrange(4,n_hwkeys):
        key = hwkeys_list[i]
        hw_n = int(key[2:])
        av_hwdata_df[key] = 1.0*av_hwdata_df[key]/hw_n

    for i in xrange(4,n_examkeys):
        key = examkeys_list[i]
        if key == 'F':
            av_examdata_df[key] = 1.0*av_examdata_df[key]/3            
        else:
            exam_n = int(key[1:])
            av_examdata_df[key] = 1.0*av_examdata_df[key]/exam_n

    print "Here we there yet?"

    if n_midterms == 2:
        print "Here we are now"
        print minmax_midtermdata_df
        sys.exit()         
        
#    print av_hwdata_df
#    print av_examdata_df


    projection_df = rawdata_df.copy()
    for i in xrange(4,n_keys):
        key = keys_list[i]
        projection_df[key] = 0
        if key in ('HW1','HW2','HW3','HW4'):
            projection_df[key] = av_hwdata_df[key]
        elif key == 'M1':
            projection_df[key] = 0.2*av_hwdata_df['HW4']+0.8*av_examdata_df['M1']
        elif key in ('HW5', 'HW6','HW7','HW8'):
            projection_df[key] = 0.2*av_hwdata_df[key]+0.8*av_examdata_df['M1']
        elif key == 'M2':
            projection_df[key] = 0.2*av_hwdata_df['HW8']+0.3*av_examdata_df['M1']
        else:
            sys.exit("Not yet implemented!")


#     print_df(projection_df)

#     print keys_list



#     # OLD VERSION
#     # empty all data fields in projection_df
#     for i in xrange(4,n_keys):
#         key = keys_list[i]
#         accumulateddata_df[key] = 0        
#         projection_df[key] = 0
#         if key == 'HW1':
#             projection_df[key] = rawdata_df[key]
#         elif key in ('HW2', 'HW3','HW4'):
#             for j in xrange(4,i+1):
#                 keytmp = keys_list[j]
#                 projection_df[key] += rawdata_df[keytmp]
#             projection_df[key] = projection_df[key]/(i-3)
#         elif key == 'M1':
#             projection_df[key] = 0.2*projection_df['HW4']+0.8*rawdata_df['M1']
#         elif key in ('HW5', 'HW6','HW7'):
#             for j in xrange(4,i+1):
#                 keytmp = keys_list[j]
#                 projection_df[key] += rawdata_df[keytmp]
#             projection_df[key] = projection_df[key]/(i-3)

    
    # open text dump file
    messagefile_name = 'messagefile_' + std_datetime_str('date') + '.txt'
    messagefile = open(messagefile_name,'w',0)

    for index in rawdata_df.index:
        tmp_str = rawdata_df.loc[index, 'email']
        messagefile.write(tmp_str + '\n')
        update_n = n_hws + n_midterms + n_final 
        tmp_str = "Grade summary and projection for CE 317 (#" + str(update_n) + ")" 
        messagefile.write(tmp_str + '\n\n')

        firstname = rawdata_df.loc[index, 'First name'].split()[0]
        if firstname == ".":
            firstname = rawdata_df.loc[index, 'Last name'].split()[0]
        
        tmp_str = "Dear " + firstname + ","
        messagefile.write(tmp_str + '\n\n')
        
        tmp_str = "I'm writing to give you a brief update on where you stand in CE 317. Here are the marks I have on record for you so far:" 
        messagefile.write(tmp_str + '\n')

#         tmp_str = str(rawdata_df.loc[index,'HW1':])
# #         tmp_str = str(rawdata_df[index, 4:])
#         print tmp_str
#         sys.exit()
#         messagefile.write(tmp_str + '\n\n')
        for i in xrange(4,n_keys):
            key = keys_list[i]
            tmp_str = key + ": " 
            if len(key) == 2:
                tmp_str += " " 
            tmp_str += " %5.1f " %(rawdata_df.iloc[index, i])
            messagefile.write(tmp_str + '\n')
        messagefile.write('\n\n')

        tmp_str = "In the following you can find the class statistics for each assignment/exam:" 
        messagefile.write(tmp_str + '\n\n')
 
        pd.options.display.float_format = '{:7.2f}'.format
        tmp_str = str(rawdata_df.loc[:,'HW1':].describe())
#         tmp_str = str(rawdata_df.describe())
        messagefile.write(tmp_str + '\n\n\n')

        tmp_str = "Based on your assignment marks, I arrived at the following grade projections:" 
        messagefile.write(tmp_str + '\n')

        for i in xrange(4,n_keys):
            key = keys_list[i]
            tmp_str = "Grade projection after " + key + ": " 
            if len(key) == 2:
                tmp_str += " " 
            tmp_str += " %5.1f " %(projection_df.iloc[index, i])
            tmp_str += "(" + percent2lettergrade(projection_df.iloc[index, i]) + ")"
            messagefile.write(tmp_str + '\n')
        messagefile.write('\n')

        if percent2lettergrade(projection_df.iloc[index, i]) == 'A':
            tmp_str = "Well done - excellent job, " + firstname + "! Keep up the good work!"  
            messagefile.write(tmp_str + '\n\n')

        tmp_str = "Note: These grade projections are based on default 5-point lettergrade brackets as well as the weights for exams and homeworks indicated in the course syllabus. " 
        tmp_str += "Your prior homework and exam averages are used as placeholders for the missing homeworks and exams, respectively. \n" 
        tmp_str += "They do NOT yet incorporate extra credit for in-class participation, nor do they consider potential adjustments to the grade brackets. \n"
        tmp_str += "I'm providing the grades after each assignment to give you an idea about your progress. "
        tmp_str += "It is worth noting that grades tend to pick up after the first midterm.\n"
        tmp_str += "Please let me know if you have any questions or concerns."
        messagefile.write(tmp_str + '\n\n')

        if opts.requestmeeting is True:
            if projection_df.iloc[index, i] < 66:
                tmp_str = firstname + ", since you are current not doing so great, I wanted to offer to have a meeting with you to see what we can do to improve things. Please let me know what you think."   
                messagefile.write(tmp_str + '\n\n\n')


        tmp_str = "Best wishes,"
        messagefile.write(tmp_str + '\n\n')

        tmp_str = "JH"
        messagefile.write(tmp_str + '\n\n\n')
        tmp_str = "------------------------------------------------------------------------------ "
        messagefile.write(tmp_str + '\n\n')
         

    messagefile.close()
    sys.exit("test 14")

# TODO compute other cases         

    print_df(projection_df)
    print_df(rawdata_df)

    


    

    
#     classsummary_df = rawdata_df
#     print_df(classsummary_df)




    # create lists of various grades
    for id in id_list:
        data_dict[id]['hw_grade_av'] = []
        data_dict[id]['overall_grade'] = []
        data_dict[id]['overall_lettergrade'] = []


    # create assignment keys list for better readability; introduce assignment keys list; note: we trade resources for readability 
    assignment_keys_list = keys_list[3:]
    n_assignment_keys = len(assignment_keys_list)

    
    # we want grades for every point during the semester, so we successively go through list of assignments and compute grade after each     
    for i in xrange(n_assignment_keys):        
        # determine number of homeworks at any point in semester
        n_hw = 0
        for key in assignment_keys_list[0:i+1]:
            if "HW" in key: n_hw +=1        
        for id in id_list:
        # distinguish different cases for grade projections, depending on where we stand in the semester
            if 'Final' in assignment_keys_list[0:i+1]:  # i.e., this is the final grade after all assignments are in 
                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
                midterm_max = max(data_dict[id]['midterm_grades'])
                midterm_min = min(data_dict[id]['midterm_grades'])
                final = data_dict[id]['final_grade']    # this is really for readability    
            elif 'M2' in assignment_keys_list[0:i+1]:
                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
                midterm_max = max(data_dict[id]['midterm_grades'])
                midterm_min = min(data_dict[id]['midterm_grades'])
                final = sum(data_dict[id]['midterm_grades'])/len(data_dict[id]['midterm_grades'])
            elif 'M1' in assignment_keys_list[0:i+1]:
                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
                midterm_max = max(data_dict[id]['midterm_grades'])
                midterm_min = min(data_dict[id]['midterm_grades'])
                final = sum(data_dict[id]['midterm_grades'])/len(data_dict[id]['midterm_grades'])    
            elif 'HW1' in assignment_keys_list[0:i+1]:
                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
                midterm_max = hw_average
                midterm_min = hw_average
                final = hw_average
            else:
                tmp_str = "Aborting due to lack of reported grades!"
                logfile.write(tmp_str + '\n')
                error_file.write(tmp_str + '\n')
                sys.exit(tmp_str)
                
            # implement grading scheme: HW: 20%, better midterm 35%, worse midterm 15%, final: 30%
            overall_grade = 0.2*hw_average + 0.35*midterm_max + 0.15*midterm_min + 0.3*final
# TODO: instead of hardwiring, we may want to build more flexibility in here 
            
            overall_lettergrade = percent2lettergrade(overall_grade)

            # add computed information to data dictionary
            data_dict[id]['hw_grade_av'].append(hw_average)
# TODO: we should take out the rounding here
            data_dict[id]['overall_grade'].append(round(overall_grade,1))
            data_dict[id]['overall_lettergrade'].append(overall_lettergrade)


#     # output for testing
#     for id in id_list:
#         print str(id) + ' ' + str(data_dict[id]['overall_grade'])+ ' ' + str(data_dict[id]['overall_lettergrade'])

    tmp_str = "...calculation of grades and grade projections finished."
    print tmp_str
    logfile.write(tmp_str + '\n')


    #################################################################################################


    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "Starting calculation of course statistics..."
    print tmp_str
    logfile.write(tmp_str + '\n')
    
#     tmp_str = "   ...computing basic distribution statistics..."
#     print tmp_str
#     logfile.write(tmp_str + '\n')

    # create lists of lists with all the overall grades
    course_overall_grade_list = []
    course_overall_lettergrade_list = []
    course_overall_grade_stat_list = []

    # iterate through all assignments
    for j in xrange(n_assignment_keys):
        course_overall_grade_list.append([])
        course_overall_lettergrade_list.append([])
        for id in id_list:
            course_overall_grade_list[j].append(data_dict[id]['overall_grade'][j])
            course_overall_lettergrade_list[j].append(data_dict[id]['overall_lettergrade'][j])
            

        stat = distribution_stat(course_overall_grade_list[j])
        course_overall_grade_stat_list.append(stat)

        course_overall_grade_stat_list[j]['letter_av'] = percent2lettergrade(course_overall_grade_stat_list[j]['av'])
        course_overall_grade_stat_list[j]['letter_median'] = percent2lettergrade(course_overall_grade_stat_list[j]['median'])
        course_overall_grade_stat_list[j]['letter_min'] = percent2lettergrade(course_overall_grade_stat_list[j]['min'])
        course_overall_grade_stat_list[j]['letter_max'] = percent2lettergrade(course_overall_grade_stat_list[j]['max'])
                
        course_overall_grade_stat_list[j]['letter_dist'] = histogram(course_overall_lettergrade_list[j])

        # TODO: here we need a proper print statement now.
        print course_overall_grade_stat_list[j]
        print 
        sys.exit("This is as far as it goes right now.")


    tmp_str = "   ...computing letter grade distribution..."
    print tmp_str
    logfile.write(tmp_str + '\n')


# perform statistics, analysis, projections
# compute current average according to grading rules 
# rank students
# identify best, worst students
# compile info for each student
# visualize trends
# add course participation into grading scheme
# test different grading schemes 


    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str

#     print str(grade_total_average) + '  ' +  grade_total_average_letter
    
    
    # rank students
    # identify best, worst students
    # note: there is no good way to sort a nested dictionary by value, so we just create an auxillary dictionary
    tmp_list = []
    for id in id_list:
        tmp_tuple = (id,data_dict[id]['grade_total'])
        tmp_list.append(tmp_tuple)

    print tmp_list
    print 
    print 

    sorted_tmp_list = sorted(tmp_list, key=itemgetter(1))
    print sorted_tmp_list
    

    

    tmp_str = "... finished."
    print tmp_str
    logfile.write(tmp_str + '\n')
    tmp_str = "------------------------------------------------------------------------------ "
    print tmp_str
    logfile.write(tmp_str + '\n')


    #################################################################################################

    # wrap up section
    tmp_str = tot_exec_time_str(time_start) + "\n" + std_datetime_str()
    print tmp_str + 3*'\n'
    logfile.write(tmp_str + 4*'\n')
    logfile.close()    
    error_file.close()
    
    # check whether error_file contains content
    chk_rmfile(opts.error_file)
        
    return 0    #successful termination of program
    
#################################################################################################

# TODO: replace with argpass 
if __name__=="__main__":
    usage_str = "usage: %prog [options] arg"
    version_str = "%prog " + SCRIPT_VERSION
    parser = OptionParser(usage=usage_str, version=version_str)    
    
    parser.add_option('--data_file', 
                      dest='data_file', 
                      type='string', 
                      help='specifies the name of the raw data file in CSV format')

    parser.add_option('--job_file', 
                      dest='job_file', 
                      type='string', 
                      help='specifies the name of the job file that specifies sets ')
# TODO: need to write a parser for the jobfile

    parser.add_option('--requestmeeting', 
                      dest='requestmeeting', 
                      action='store_true', 
                      default=False, 
                      help='specifies the a meeting is requested in the student email')


    # Generic options 
    parser.add_option('--print_level', 
                      dest='print_level', 
                      type='int', 
                      default=2, 
                      help='specifies the print level for on screen and the logfile [default: %default]')
        
    # specify log files 
    parser.add_option('--logfile', 
                      dest='logfile', 
                      type='string', 
                      default='grademaster.log',  
                      help='specifies the name of the log-file [default: %default]')

    parser.add_option('--error_file', 
                      dest='error_file', 
                      type='string', 
                      default='grademaster.err',  
                      help='specifies the name of the error-file [default: %default]')

    opts, args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        sys.exit("You tried to run grademaster without options.")
    main(opts,sys.argv)
    
else:
    sys.exit("Sorry, must run as driver...")
    