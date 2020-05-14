#!/usr/bin/env python

SCRIPT_NAME = "grademaster"
SCRIPT_VERSION = "v0.4.0"
REVISION_DATE = "2020-05-07"
AUTHOR = """
Johannes Hachmann (hachmann@buffalo.edu) with contributions by:
   Mojtaba Haghighatlari (jobfile, Pandas dataframe)
   Nitin Murthy (Prediction algorithm, argparse conversion, Python 3 upgrade)
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
# v0.3.0  (2019-11-25): revamp
# v0.4.0  (2020-05-07): Python 3 compatibility, Prediction section build, code cleanups and transition to argparse.

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
# Change: Done
import argparse
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

def percent2lettergrade(grade,gradescheme):
    """(percent2lettergrade):
        This function converts grades into lettergrades according to a given conversion scheme.
    """
# TODO: this is the place to realize different grading-schemes in a more general fashion
    if gradescheme.lower() == 'absolute':
        if round(grade) >= 96:
            return 'A'
        elif round(grade) >= 91:
            return 'A-'
        elif round(grade) >= 86:
            return 'B+'
        elif round(grade) >= 81:
            return 'B'
        elif round(grade) >= 76:
            return 'B-'
        elif round(grade) >= 71:
            return 'C+'
        elif round(grade) >= 66:
            return 'C'
        elif round(grade) >= 61:
            return 'C-'
        elif round(grade) >= 56:
            return 'D+'
        elif round(grade) >= 51:
            return 'D'
        else:
            return 'F'
    elif gradescheme.lower() == 'percentile': #source for data: https://blog.prepscholar.com/gpa-conversion
        if round(grade) >= 93:
            return 'A'
        elif round(grade) >= 90:
            return 'A-'
        elif round(grade) >= 87:
            return 'B+'
        elif round(grade) >= 83:
            return 'B'
        elif round(grade) >= 80:
            return 'B-'
        elif round(grade) >= 77:
            return 'C+'
        elif round(grade) >= 73:
            return 'C'
        elif round(grade) >= 70:
            return 'C-'
        elif round(grade) >= 67:
            return 'D+'
        elif round(grade) >= 65:
            return 'D'
        else:
            return 'F'
    elif gradescheme.lower() == 'gpa': #source for data: https://blog.prepscholar.com/gpa-conversion
        if grade >= 4:
            return 'A'
        elif grade >= 3.7:
            return 'A-'
        elif grade >= 3.3:
            return 'B+'
        elif grade >= 3:
            return 'B'
        elif grade >= 2.7:
            return 'B-'
        elif grade >= 2.3:
            return 'C+'
        elif grade >= 2:
            return 'C'
        elif grade >= 1.7:
            return 'C-'
        elif grade >= 1.3:
            return 'D+'
        elif grade >= 1:
            return 'D'
        else:
            return 'F'

    else:
        return("We're sorry, but that grading scheme isn't defined.")
###################################################################################################

# TODO: we should really use a library/module for this rather than coding it by hand (this is just for exercise)
#Change: Done. The one function call now utilizes numpy.median()
#def median_func(val_list):
#    """(median function):
#        Takes a list of values and returns the median.
#    """
#    val_list = sorted(val_list)
#    # different cases
#    # empty list
#    if len(val_list) < 1:
#        return None
#    # list with odd numbers of values
#    if len(val_list) %2 == 1:
#        return val_list[((len(val_list)+1)/2)-1]
#    # list with even numbers of values
#    if len(val_list) %2 == 0:
#        return float(sum(val_list[(len(val_list)/2)-1:(len(val_list)/2)+1]))/2.0

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

        tmp_val_list = np.array(val_list)
        median = np.median(tmp_val_list)

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

def strucheck(stru,key_list):
    if stru not in key_list[0:4]:
        tmp_str = "   ..."+stru+" missing in data structure!"
        print(tmp_str)
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        tmp_str = "Aborting due to invalid data structure!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)



###################################################################################################

# TODO: this should potentially go into a library
#Change: This is deprecated with Python 3, replacing it with print().
#def print_df(dataframe):
#    """(print_df):
#        Prints an entire Pandas DataFrame.
#    """
#
#    pd.set_option('display.max_rows', len(dataframe))
#    print(dataframe)
#    pd.reset_option('display.max_rows')
#    return

###################################################################################################

def main(args,commline_list):
    """(main):
        Driver of the grademaster script.
    """
    time_start = time.time()

    # now the standard part of the script begins
    logfile = open(args.logfile,'a')
    error_file = open(args.error_file,'a')

    banner(logfile, SCRIPT_NAME, SCRIPT_VERSION, REVISION_DATE, AUTHOR, DESCRIPTION)

    # give out options of this run
    print_invoked_opts(logfile,args,commline_list)

    home_dir = os.getcwd()

    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    #################################################################################################
    # read in the CSV file with the raw data of grades

    # make a logfile entry and screen entry so that we know where we stand
    tmp_str = "Starting data acquisition..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    # check that file exists, get filename from optparse
    if args.data_file is None or os.path.getsize(args.data_file) == 0:
        tmp_str = "... data file not specified/empty!"
        print(tmp_str)
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')

        tmp_str = "Aborting due to missing data file!"
        logfile.write(tmp_str + '\n')
        error_file.write(tmp_str + '\n')
        sys.exit(tmp_str)
# TODO: This should better be done by exception handling


    tmp_str = "   ...reading in data..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    std_filename = args.data_file.replace('.csv','')    #Filename prefix for outputs
    os.mkdir(std_filename)
    std_filename = std_filename+'/'
    # open CSV file with raw data
    rawdata_df = pd.read_csv(args.data_file)
    print(rawdata_df)
    #sys.exit("print(rawdata_df)")

    tmp_str = "   ...cleaning data structure..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    # remove empty entries
    for i in rawdata_df.columns:
        if 'Unnamed'in i:
            rawdata_df = rawdata_df.drop(i,1)
    rawdata_df = rawdata_df.dropna(how='all')
#    print(rawdata_df)


    tmp_str = "   ...identify keys..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    # read top line of data file, which defines the keys
    keys_list = list(rawdata_df.columns)
    n_keys = len(keys_list)
#    print keys_list

    #Ensuring all scores are between 0 and 100
    for i in range(4,n_keys):
        rawdata_df[keys_list[i]] = rawdata_df[keys_list[i]].clip(lower=0,upper=100)


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
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    # check that the standard keys are amongst the first three keys, because that's all we have implemented so far


#    if "Last name" not in keys_list[0:4]:
#        tmp_str = "   ...'Last name' missing in data structure!"
#        print(tmp_str)
#        logfile.write(tmp_str + '\n')
#        error_file.write(tmp_str + '\n')
#    elif "First name" not in keys_list[0:4]:
#        tmp_str = "   ...'First name' missing in data structure!"
#        print(tmp_str)
#        logfile.write(tmp_str + '\n')
#        error_file.write(tmp_str + '\n')
#    elif "Student ID" not in keys_list[0:4]:
#        tmp_str = "   ...'Student ID' missing in data structure!"
#        print(tmp_str)
#        logfile.write(tmp_str + '\n')
#        error_file.write(tmp_str + '\n')
#    elif "email" not in keys_list[0:4]:
#        tmp_str = "   ...'email' missing in data structure!"
#        print(tmp_str)
#        logfile.write(tmp_str + '\n')
#        error_file.write(tmp_str + '\n')

    strucheck('Last name',keys_list)
    strucheck('First name', keys_list)
    strucheck('Student ID', keys_list)
    strucheck('email', keys_list)

    # check if all the grades are in float type (not object)
    for i in keys_list[4:]:
        if rawdata_df[i].dtypes == object:
            tmp_str = "Aborting due to unknown grade format in column %s!" %i
            logfile.write(tmp_str + '\n')
            error_file.write(tmp_str + '\n')
            sys.exit(tmp_str)


    course_id = str(input('Enter the course number: '))

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
#         for i in range(n_keys):    # note that we use range instead of range
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


    # some bookkeeping on where we stand in the semester
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
    print(tmp_str)
    logfile.write(tmp_str + '\n')


    #################################################################################################


    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "Summary of acquired data for course "+course_id+":"
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    tmp_str = "   Number of students:  " + str(len(rawdata_df))
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of homeworks: " + str(n_hws)
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of midterms:  " + str(n_midterms)
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "   Number of finals:    " + str(n_final)
    print(tmp_str)
    logfile.write(tmp_str + '\n')
# TODO: this should be better formatted

    #################################################################################################

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
#    hwdata_df = rawdata_df.copy()
#    examdata_df = rawdata_df.copy()
    # empty all data fields in projection_df
#    hwdata_df['Final'] = 0
#    for i in range(4,n_keys):
#        key = keys_list[i]
#        if 'HW' in key:
#            examdata_df.drop(key, axis=1, inplace=True)
#        elif key in ('M1', 'M2','F'):
#            hwdata_df.drop(key, axis=1, inplace=True)

#    print hwdata_df
#    print examdata_df


#    hwkeys_list = list(hwdata_df.columns)
#    n_hwkeys = len(hwkeys_list)

#    examkeys_list = list(examdata_df.columns)
#    n_examkeys = len(examkeys_list)

#    acc_hwdata_df = hwdata_df.copy()
#    acc_examdata_df = examdata_df.copy()

#    for i in range(4,n_hwkeys):
#        key = hwkeys_list[i]
#        if key == 'HW1':
#            continue
#        else:
#            prevkey = hwkeys_list[i-1]
#            acc_hwdata_df[key] += acc_hwdata_df[prevkey]

#    for i in range(4,n_examkeys):
#        key = examkeys_list[i]
#        if key == 'M1':
#            continue
#        else:
#            prevkey = examkeys_list[i-1]
#            acc_examdata_df[key] += acc_examdata_df[prevkey]

#    print acc_hwdata_df
#    print acc_examdata_df

#    av_hwdata_df = acc_hwdata_df.copy()
#    av_examdata_df = acc_examdata_df.copy()
#    minmax_midtermdata_df = examdata_df.copy()

#    for i in range(4,n_hwkeys):
#        key = hwkeys_list[i]
        #hw_n = int(key[2:])
#        av_hwdata_df[key] = 1.0*av_hwdata_df[key]/n_hws

#    for i in range(4,n_examkeys):
#        key = examkeys_list[i]
#        if key == 'F':
#            av_examdata_df[key] = 1.0*av_examdata_df[key]/3
#        else:
            #exam_n = int(key[1:])
#            av_examdata_df[key] = 1.0*av_examdata_df[key]/(n_midterms)

#    print("Are we there yet?")

#    if n_midterms == 2:
#        print("Here we are now")
#        print(minmax_midtermdata_df)
#        print(examdata_df)
#        print(acc_examdata_df)
#        print(av_examdata_df)
#        print(hwdata_df)
#        print(acc_hwdata_df)
#        print(av_hwdata_df)
#        sys.exit()

#    print av_hwdata_df
#    print av_examdata_df


#    for i in range(4,n_keys):
#        key = keys_list[i]
#        projection_df[key] = 0
#        if key in ('HW1','HW2','HW3','HW4'):
#            projection_df[key] = av_hwdata_df[key]
#        elif key == 'M1':
#            projection_df[key] = 0.2*av_hwdata_df['HW4']+0.8*av_examdata_df['M1']
#        elif key in ('HW5', 'HW6','HW7','HW8'):
#            projection_df[key] = 0.2*av_hwdata_df[key]+0.8*av_examdata_df['M1']
#        elif key == 'M2':
#            projection_df[key] = 0.2*av_hwdata_df['HW8']+0.3*av_examdata_df['M1']
#        else:
#            sys.exit("Not yet implemented!")


# I've moved the course statistics section above the prediction section as I feel this is required,
# while the prediction section automatically quits if the course is completed and won't execute the sections below it.

#Course Statistics Section Redux
#Rewriting section to use dataframes
    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "Starting calculation of course statistics..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)

    hw_average = 0
    midterm_max = 0
    midterm_min = 0
    final = 0

    hwdata_df = rawdata_df.copy()
    examdata_df = rawdata_df.copy()

    #Removing student ID to help with mean calculation
    hwdata_df.drop('Student ID', axis=1, inplace=True)
    examdata_df.drop('Student ID', axis=1, inplace=True)

    #hwdata_df['Final'] = 0
    for i in range(4,n_keys):
        key = keys_list[i]
        if 'HW' in key:
            examdata_df.drop(key, axis=1, inplace=True)
        elif key in ('M1', 'M2'):
            hwdata_df.drop(key, axis=1, inplace=True)
        elif 'F' in key:
            hwdata_df.drop(key, axis=1, inplace=True)
            examdata_df.drop(key,axis=1,inplace=True)

#    print(hwdata_df)
#    print(examdata_df)


    hwkeys_list = list(hwdata_df.columns)
    n_hwkeys = len(hwkeys_list)

    examkeys_list = list(examdata_df.columns)
    n_examkeys = len(examkeys_list)

    hwdata_df = hwdata_df.assign(mean=hwdata_df.mean(axis=1,numeric_only=True),std_dev=hwdata_df.std(axis=1,numeric_only=True))
    examdata_df = examdata_df.assign(mean=examdata_df.mean(axis=1,numeric_only=True),std_dev=examdata_df.std(axis=1,numeric_only=True))
#    print(hwdata_df)
    if 'M1' in examkeys_list and 'M2' in examkeys_list:
        examdata_df = examdata_df.assign(max=examdata_df.max(axis=1),min=examdata_df[['M1','M2']].min(axis=1))
    elif 'M1' in examkeys_list and 'M2' not in examkeys_list:
        pass

    #participation grade: Just a randomized distribution with max of 5 and min of 0
    participate = np.random.uniform(low=0, high=5, size=(len(list(hwdata_df.index))))
    hwdata_df['Participation'] = participate
    hwdata_df = hwdata_df.assign(adj_mean = hwdata_df['mean']+hwdata_df['Participation'])
    hwdata_df['adj_mean'] = hwdata_df['adj_mean'].clip(lower=0,upper=100)

    tmp_str = "Course status:\n"
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    if n_midterms > 0 or n_final > 0:
        tmp_str = "Exam data:\n"
        print(tmp_str)
        logfile.write(tmp_str + '\n')
        print(examdata_df)
    tmp_str = "Homework data:\n"
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    print(hwdata_df)
    plt.plot(hwdata_df['mean'])
    plt.savefig(std_filename+'grade_plot.png')

    finaldata_df = rawdata_df.copy()
    finaldata_df = finaldata_df.assign(hw_mean=hwdata_df['adj_mean'],exam_mean=examdata_df['mean'])

    # Final grading scheme: HW: 20%, better midterm 35%, worse midterm 15%, final: 30%
    hw_grad = 0.2
    midmax_grad = 0.35
    midmin_grad = 0.15
    fin_grad = 0.3
    #overall_grade = hw_grad*hw_average + midmax_grad*midterm_max + midmin_grad*midterm_min + fin_grad*final
    #overall_lettergrade = percent2lettergrade(overall_grade,'absolute')
    # TODO: instead of hardwiring, we may want to build more flexibility in here
    # Change: Done.

    print("Grading scheme:")
    print("Homework: "+str(hw_grad*100)+'%')
    print("Highest midterm: "+str(midmax_grad*100)+'%')
    print("Lowest midterm: "+str(midmin_grad*100)+'%')
    print("Final: "+str(fin_grad*100)+'%')

    #Now to write this into the database:
    if n_final > 0:
        finaldata_df = finaldata_df.assign(overall_mean=hwdata_df['adj_mean']*hw_grad+ midmax_grad*examdata_df['max'] + midmin_grad*examdata_df['min'] + fin_grad*rawdata_df['Final'])
    elif n_midterms == 2:    #If there's no final yet, compensate the grading scheme by making the sum of the remaining grades 1.
        grad_adj = 1/(hw_grad+midmax_grad+midmin_grad)
        hw_grad = grad_adj*hw_grad
        midmax_grad = grad_adj*midmax_grad
        midmin_grad = grad_adj*midmin_grad
        finaldata_df = finaldata_df.assign(overall_mean=hwdata_df['adj_mean']*hw_grad+ midmax_grad*examdata_df['max'] + midmin_grad*examdata_df['min'])
    elif n_midterms == 1:    #If only one midterm has happened, all midterm weights are applied to this.
        grad_adj = 1/(hw_grad+midmax_grad+midmin_grad)
        hw_grad = grad_adj*hw_grad
        midmax_grad = grad_adj*midmax_grad
        midmin_grad = grad_adj*midmin_grad
        finaldata_df = finaldata_df.assign(overall_mean=hwdata_df['adj_mean']*hw_grad+ midmax_grad*examdata_df['mean'] + midmin_grad*examdata_df['mean'])
    else:                   #If there have been no exams yet, just use the homework average without weights.
        finaldata_df = finaldata_df.assign(overall_mean=hwdata_df['adj_mean'])
    finaldata_df['grade']=0
    for i in finaldata_df.index:
        finaldata_df.loc[i,'grade'] = percent2lettergrade(finaldata_df.loc[i,'overall_mean'],'absolute')
    finaldata_df = finaldata_df.sort_values(by = 'overall_mean', ascending=False)
    finaldata_df.dropna(axis=1)

    tmp_str = "Sorted and graded list:\n"
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    print(finaldata_df)

    tmp_str = "Congratulations to the toppers!"
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    if n_hws < 5:
        tmp_str = "To those with lower grades, don't lose heart! There's still time to make up!"
        print(tmp_str)
        logfile.write(tmp_str + '\n')

    tmp_str = "Exporting sorted data to CSV."
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    finaldata_df.to_csv(path_or_buf=std_filename+'Results.csv'+std_datetime_str('date'))


    # add computed information to data dictionary
#    data_dict[id]['hw_grade_av'].append(hw_average)
# TODO: we should take out the rounding here
#    data_dict[id]['overall_grade'].append(round(overall_grade,3))
#    data_dict[id]['overall_lettergrade'].append(overall_lettergrade)


#     # output for testing
#     for id in id_list:
#         print str(id) + ' ' + str(data_dict[id]['overall_grade'])+ ' ' + str(data_dict[id]['overall_lettergrade'])


    #################################################################################################

#    Deprecated sorting section
#     tmp_str = "   ...computing basic distribution statistics..."
#     print tmp_str
#     logfile.write(tmp_str + '\n')

    # create lists of lists with all the overall grades
#    course_overall_grade_list = []
#    course_overall_lettergrade_list = []
#    course_overall_grade_stat_list = []

    # iterate through all assignments
#    for j in range(n_assignment_keys):
#        course_overall_grade_list.append([])
#        course_overall_lettergrade_list.append([])
#        for id in id_list:
#            course_overall_grade_list[j].append(data_dict[id]['overall_grade'][j])
#            course_overall_lettergrade_list[j].append(data_dict[id]['overall_lettergrade'][j])


#        stat = distribution_stat(course_overall_grade_list[j])
#        course_overall_grade_stat_list.append(stat)

#        course_overall_grade_stat_list[j]['letter_av'] = percent2lettergrade(course_overall_grade_stat_list[j]['av'],'absolute')
#        course_overall_grade_stat_list[j]['letter_median'] = percent2lettergrade(course_overall_grade_stat_list[j]['median'],'absolute')
#        course_overall_grade_stat_list[j]['letter_min'] = percent2lettergrade(course_overall_grade_stat_list[j]['min'],'absolute')
#        course_overall_grade_stat_list[j]['letter_max'] = percent2lettergrade(course_overall_grade_stat_list[j]['max'],'absolute')

#        course_overall_grade_stat_list[j]['letter_dist'] = histogram(course_overall_lettergrade_list[j])

        # TODO: here we need a proper print statement now.
#        print(course_overall_grade_stat_list[j])
#        print()
#        sys.exit("This is as far as it goes right now.")


#    tmp_str = "   ...computing letter grade distribution..."
#    print(tmp_str)
#    logfile.write(tmp_str + '\n')


# perform statistics, analysis, projections
# compute current average according to grading rules
# rank students
# identify best, worst students
# compile info for each student
# visualize trends
# add course participation into grading scheme
# test different grading schemes


#     print str(grade_total_average) + '  ' +  grade_total_average_letter


    # rank students
    # identify best, worst students
    # note: there is no good way to sort a nested dictionary by value, so we just create an auxillary dictionary
#    tmp_list = []
#    for id in id_list:
#        tmp_tuple = (id,data_dict[id]['grade_total'])
#        tmp_list.append(tmp_tuple)

#    print(tmp_list)

#    sorted_tmp_list = sorted(tmp_list, key=itemgetter(1))
#    print(sorted_tmp_list)

# Prediction Section 1.0 - A simple gradient based predictor, can be replaced by a better prediction algorithm later.
# Director's Note:  I thought of using some kind of regression, but there's too little data to make any sensible pattern.
#                   I'm most likely doing something wrong, this can be fixed in future versions.

    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "Starting calculation of grade projections..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    projection_df = rawdata_df.copy()

    n_hws_total = 10
    n_midterms_total = 2
    n_finals_total = 1
    #emptyrows, emptycolumns = np.where(pd.isnull(projection_df))
    emptykeys = []
#populating prediction db with current data

    for i in range(4,n_keys):
      key = keys_list[i]
      projection_df[key] = 0
      projection_df[key] = rawdata_df[key]

#Prediction Algorithm:
#   1. Check which columns are missing. (Complete!)
#   2. Determine gradient based on final 3 entries. (Could be refined in a future version)
#   3. Predict scores based on no. of assignments completed and gradient. (no. of assignments predicted <= no. of assignments completed)

#   1. Generating list of empty columns
    for i in range(n_hws+1, n_hws_total+1):
        emptykeys.append('HW'+str(i))
    for i in range(len(emptykeys)):
        if emptykeys[i-1] == 'HW4' and emptykeys[i]!='M1' or emptykeys[i]=='HW5' and emptykeys[i-1]!='M1':
            emptykeys.insert(i,'M1')
        if emptykeys[i-1] == 'HW8' and emptykeys[i]!='M2' or emptykeys[i]=='HW9' and emptykeys[i-1]!='M2':
            emptykeys.insert(i,'M2')
    if emptykeys[:0] != 'F':
        emptykeys.append('F')
    #print(emptykeys)
    endkey = keys_list[n_keys-1]
    #print(endkey)
    if n_keys < 6:
        tmp_str = "Too few datapoints!"
        logfile.write(tmp_str + '\n')
        sys.exit(tmp_str)
    elif n_keys >= 17:
        tmp_str = "Semester is already over, just look at the data!"
        logfile.write(tmp_str + '\n')
        sys.exit(tmp_str)


#   2. Gradient determination

    gradient_samplesize = min(3,n_keys-4) # this setting is a lot of fun, tweak to check accuracy
    for i in range(n_keys-gradient_samplesize,n_keys):
        projection_df['Gradient'+str(i)] = projection_df[keys_list[i]]-projection_df[keys_list[i-1]]


    proj_keys_list = list(projection_df.columns)
    proj_n_keys = len(proj_keys_list)

    col_tmp = projection_df.loc[:, 'Gradient'+str(n_keys-gradient_samplesize):'Gradient'+str(n_keys-1)]
    projection_df['MeanGradient'] = col_tmp.mean(axis=1)

#    print(projection_df)    #Gradients included, uncomment for diagnostics

    count=0
#   3. Score prediction and entry
    j=n_keys
    #print(n_keys+(n_keys-4+count), 4+n_hws_total+n_midterms_total+n_finals_total+count,j)
    #print(range(n_keys,min(n_keys+(n_keys-4+count),4+n_hws_total+n_midterms_total+n_finals_total+count)))
    while j in range(n_keys,min(n_keys+(n_keys-4+count),4+n_hws_total+n_midterms_total+n_finals_total+count)):
        #print(n_keys+(n_keys-4+count), 4+n_hws_total+n_midterms_total+count, j)
        if 'Gradient' not in proj_keys_list[j-1]:
            projection_df.insert(j, emptykeys[j-n_keys-count], projection_df[proj_keys_list[j-1]]+projection_df['MeanGradient'])
            proj_keys_list.append(emptykeys[j-n_keys-count])
        #    print(emptykeys[j-n_keys-1-count])
        #    print(proj_keys_list[j-1])
        else:
            count+=1

        j+=1

    proj_keys_list = list(projection_df.columns)
    proj_n_keys = len(proj_keys_list)

    #Conditioning the projection database:
    for i in range(4,proj_n_keys):
        if 'Gradient' in proj_keys_list[i]:
            projection_df.drop(proj_keys_list[i], axis=1, inplace=True)
        else:
            projection_df[proj_keys_list[i]] = projection_df[proj_keys_list[i]].clip(0,100)
            projection_df[proj_keys_list[i]] = projection_df[proj_keys_list[i]].astype(int)

    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "Projected scores"
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    print(projection_df)    #Final prediction database
    projection_df.to_csv(path_or_buf=std_filename+'Predictions.csv')
    proj_keys_list = list(projection_df.columns)
    proj_n_keys = len(proj_keys_list)

#     print keys_list



#     # OLD VERSION
#     # empty all data fields in projection_df
#     for i in range(4,n_keys):
#         key = keys_list[i]
#         accumulateddata_df[key] = 0
#         projection_df[key] = 0
#         if key == 'HW1':
#             projection_df[key] = rawdata_df[key]
#         elif key in ('HW2', 'HW3','HW4'):
#             for j in range(4,i+1):
#                 keytmp = keys_list[j]
#                 projection_df[key] += rawdata_df[keytmp]
#             projection_df[key] = projection_df[key]/(i-3)
#         elif key == 'M1':
#             projection_df[key] = 0.2*projection_df['HW4']+0.8*rawdata_df['M1']
#         elif key in ('HW5', 'HW6','HW7'):
#             for j in range(4,i+1):
#                 keytmp = keys_list[j]
#                 projection_df[key] += rawdata_df[keytmp]
#             projection_df[key] = projection_df[key]/(i-3)


    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "Starting automated message generation.\n"
    print(tmp_str)
    logfile.write(tmp_str + '\n')

    # open text dump file
    messagefile_name = std_filename+'_messagefile_' + std_datetime_str('date') + '.txt'
    messagefile = open(messagefile_name,'w')

    for index in projection_df.index:
        tmp_str = rawdata_df.loc[index, 'email']
        messagefile.write(tmp_str + '\n')
        update_n = n_hws + n_midterms + n_final
        tmp_str = "Grade summary and projection for "+course_id+" (#" + str(update_n) + ")"
        messagefile.write(tmp_str + '\n\n')

        firstname = rawdata_df.loc[index, 'First name'].split()[0]
        if firstname == ".":
            firstname = rawdata_df.loc[index, 'Last name'].split()[0]

        tmp_str = "Dear " + firstname + ","
        messagefile.write(tmp_str + '\n\n')

        tmp_str = "I'm writing to give you a brief update on where you stand in "+course_id+". Here are the marks I have on record for you so far:"
        messagefile.write(tmp_str + '\n')

#         tmp_str = str(rawdata_df.loc[index,'HW1':])
# #         tmp_str = str(rawdata_df[index, 4:])
#         print tmp_str
#         sys.exit()
#         messagefile.write(tmp_str + '\n\n')
        for i in range(4,n_keys):
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

        for i in range(n_keys,proj_n_keys):
            key = proj_keys_list[i]
            if 'Gradient' not in key:
                tmp_str = "Grade projection after " + key + ": "
                if len(key) == 2:
                    tmp_str += " "
                tmp_str += " %5.1f " %(projection_df.iloc[index, i])
                tmp_str += "(" + percent2lettergrade(projection_df.iloc[index, i],'absolute') + ")"
                messagefile.write(tmp_str + '\n')
        messagefile.write('\n')

        if percent2lettergrade(projection_df.iloc[index, i],'absolute') == 'A':
            tmp_str = "Well done - excellent job, " + firstname + "! Keep up the good work!"
            messagefile.write(tmp_str + '\n\n')

        tmp_str = "Note: These grade projections are based on default 5-point lettergrade brackets as well as the weights for exams and homeworks indicated in the course syllabus. "
        tmp_str += "Your prior homework and exam averages are used as placeholders for the missing homeworks and exams, respectively. \n"
        tmp_str += "They do NOT yet incorporate extra credit for in-class participation, nor do they consider potential adjustments to the grade brackets. \n"
        tmp_str += "I'm providing the grades after each assignment to give you an idea about your progress. "
        tmp_str += "It is worth noting that grades tend to pick up after the first midterm.\n"
        tmp_str += "Please let me know if you have any questions or concerns."
        messagefile.write(tmp_str + '\n\n')

        if args.requestmeeting is True:
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
    tmp_str = "Message file successfully generated. Check data directory.\nClosing..."
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')
#    sys.exit("test 14")


    tmp_str = "...calculation of grades and grade projections finished."
    print(tmp_str)
    logfile.write(tmp_str + '\n')
    tmp_str = "------------------------------------------------------------------------------ "
    print(tmp_str)
    logfile.write(tmp_str + '\n')

# Old course statistics section

# TODO compute other cases



#     classsummary_df = rawdata_df
#     print(classsummary_df)

#    id_list = []
#    data_dict = []
#    data_dict.append([])
#    for i in range(n_keys):
#        if 'id' in keys_list[i].lower():
#            id_list = rawdata_df[keys_list[i]].tolist()


    # create lists of various grades
#    for id in id_list:
#        data_dict[id]['hw_grade_av'] = []
#        data_dict[id]['overall_grade'] = []
#        data_dict[id]['overall_lettergrade'] = []


    # create assignment keys list for better readability; introduce assignment keys list; note: we trade resources for readability
#    assignment_keys_list = keys_list[3:]
#    n_assignment_keys = len(assignment_keys_list)


    # we want grades for every point during the semester, so we successively go through list of assignments and compute grade after each
#    for i in range(n_assignment_keys):
        # determine number of homeworks at any point in semester
#        n_hw = 0
#        for key in assignment_keys_list[0:i+1]:
#            if "HW" in key: n_hw +=1
#        for id in id_list:
#        # distinguish different cases for grade projections, depending on where we stand in the semester
#            if 'Final' in assignment_keys_list[0:i+1]:  # i.e., this is the final grade after all assignments are in
#                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
#                midterm_max = max(data_dict[id]['midterm_grades'])
#                midterm_min = min(data_dict[id]['midterm_grades'])
#                final = data_dict[id]['final_grade']    # this is really for readability
#            elif 'M2' in assignment_keys_list[0:i+1]:
#                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
#                midterm_max = max(data_dict[id]['midterm_grades'])
#                midterm_min = min(data_dict[id]['midterm_grades'])
#                final = sum(data_dict[id]['midterm_grades'])/len(data_dict[id]['midterm_grades'])
#            elif 'M1' in assignment_keys_list[0:i+1]:
#                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
#                midterm_max = max(data_dict[id]['midterm_grades'])
#                midterm_min = min(data_dict[id]['midterm_grades'])
#                final = sum(data_dict[id]['midterm_grades'])/len(data_dict[id]['midterm_grades'])
#            elif 'HW1' in assignment_keys_list[0:i+1]:
#                hw_average = sum(data_dict[id]['hw_grades'][0:n_hw])/len(data_dict[id]['hw_grades'][0:n_hw])
#                midterm_max = hw_average
#                midterm_min = hw_average
#                final = hw_average
#            else:
#                tmp_str = "Aborting due to lack of reported grades!"
#                logfile.write(tmp_str + '\n')
#                error_file.write(tmp_str + '\n')
#                sys.exit(tmp_str)
    #################################################################################################

    # wrap up section
    tmp_str = tot_exec_time_str(time_start) + "\n" + std_datetime_str()
    print(tmp_str + 3*'\n')
    logfile.write(tmp_str + 4*'\n')
    logfile.close()
    error_file.close()

    # check whether error_file contains content
    chk_rmfile(args.error_file)

    return 0    #successful termination of program

#################################################################################################

# TODO: replace with argpass
if __name__=="__main__":
    usage_str = "usage:  [options] arg"
    version_str = SCRIPT_VERSION
    #parser = OptionParser(usage=usage_str, version=version_str)
    parser = argparse.ArgumentParser(description = version_str + '\n' + usage_str )
    parser.add_argument('--data_file',
                      dest='data_file',
                      type=str,
                      help='specifies the name of the raw data file in CSV format')

#    parser.add_argument('--job_file',
#                      dest='job_file',
#                      type=str,
#                      default = 'grademaster.job',
#                      help='specifies the name of the job file that specifies sets ')
# TODO: need to write a parser for the jobfile

    parser.add_argument('--requestmeeting',
                      dest='requestmeeting',
                      action='store_true',
                      default=False,
                      help='specifies if a meeting is requested in the student email')


    # Generic options
#    parser.add_argument('--print_level',
#                      dest='print_level',
#                      type=int,
#                      default=2,
#                      help='specifies the print level for on screen and the logfile')# [default: %default]')

    # specify log files
    parser.add_argument('--logfile',
                      dest='logfile',
                      type=str,
                      default='grademaster.log',
                      help='specifies the name of the log-file')# [default: %default]')

    parser.add_argument('--error_file',
                      dest='error_file',
                      type=str,
                      default='grademaster.err',
                      help='specifies the name of the error-file')# [default: %default]')

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        sys.exit("You tried to run grademaster without options.")
    main(args,sys.argv)

else:
    sys.exit("Sorry, must run as driver...")
