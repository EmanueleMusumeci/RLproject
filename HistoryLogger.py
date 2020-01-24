from collections import namedtuple
import os
from os.path import abspath
import glob
import logging
import inspect
import errno

import datetime

HistoryValues = namedtuple("HistoryValues","mean_reward max_reward kl_div value_loss policy_loss")
class HistoryLogger:
    def __init__(self, name="", log_directory="TRPO_project/Models", log_filename=""):

        if log_filename=="":
            if name == None:
                self.name = "UnnamedLog"
            else:
                self.name = name

            #Create directory if it doesn't exist
            try:
                os.makedirs(log_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            log_directory=abspath(log_directory)
            folder = os.fsencode(log_directory)
            #filenames = []

            #Log names will be like "name.number.log" so we first check 
            #if there's some file with the same name as our logger, in this case
            #we get the latest log number and eventually we create a log file 

            #name.number+1.log
            
            last_log =""

            #Store current path to restore it later 
            current_path = os.getcwd()
            os.chdir(folder)
            files = filter(os.path.isfile, os.listdir(folder))
            files = [os.path.join(folder, f) for f in files] # add path to each file
            #order files in reversed order
            files.sort(key=lambda x: os.path.getmtime(x))
            #turn from bytestring to string
            files = [f.decode("utf-8") for f in files]

            #Change back to previous path
            os.chdir(current_path)

            for file in files:
                filename, file_extension = os.path.splitext(file)
                print(filename)
                if file_extension==".hst":
                    filename = filename.split("/")
                    filename = filename[len(filename)-1]
                    if filename.startswith(self.name):
                        last_log = file
    
            #last_log will either contain ".<number+1>" or "" if there's no log file with
            # same name 
            #we found a log file with our same name
            if last_log!="":
                last_log = last_log.split(".")
                if len(last_log)>2:    
                    #we get the middle section containing the log number
                    print(last_log)
                    #increment last log number and turn it to string
                    last_log = "."+str(int(last_log[len(last_log)-2])+1)
                    print(last_log)
                #In this case the last log file with same name was the first one
                else:
                    last_log = ".1"
            self.log_filename = log_directory + "/" + self.name + last_log + ".hst"
        else:
            self.log_filename = log_filename

        self.history = []

        #Write to file logger initializations
        with open(self.log_filename,"w+") as log_file:
            log_file.close()
    
    def log_history(self, mean_reward, max_reward, kl_div, value_loss, policy_loss):
        history_values = HistoryValues(mean_reward,max_reward,kl_div,value_loss,policy_loss)
        log_string=str(mean_reward)
        log_string+=","+str(max_reward)
        log_string+=","+str(kl_div)
        log_string+=","+str(value_loss)
        log_string+=","+str(policy_loss)
        self.history.append(history_values)
        with open(self.log_filename,"a+") as log_file:
            log_file.write(log_string+"\n")
            log_file.close()
    
    def load_history(self, filename=""):
        if filename=="": filename = self.log_filename
        self.history = []
        with open(filename, "r") as history_file:
            lines = history_file.readlines()
            for line in lines:
                line = line.split(",")
                assert(len(line)==5), "Malformed history"
                self.history.append(HistoryValues(line[0],line[1],line[2],line[3],line[4]))
    
#    #TODO: function to plot saved training history
    #def plot_history():

if __name__ == "__main__":
    import numpy as np
    env_name = "MountainCar-v0"
    hl = HistoryLogger(name=env_name)
    meanr = np.arange(10, dtype="float64")
    maxr = np.arange(10, dtype="float64")
    meankl = np.arange(10, dtype="float64")
    value_loss = np.arange(10, dtype="float64")
    policy_loss = np.arange(10, dtype="float64")
    for i in range(10):
        hl.log(meanr[i],maxr[i],meankl[i],value_loss[i],policy_loss[i])
    
    hl2 = HistoryLogger(name=env_name)
    hl2.load_history()