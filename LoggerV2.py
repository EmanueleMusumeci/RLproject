from collections import namedtuple
import os
from os.path import abspath
import glob
import logging
import inspect
import errno
import re

import datetime

import matplotlib.pyplot as plt
import json
import numpy as np

class Logger:
    def __init__(self, name=None, log_directory="", log_filename="", history_log_filename="", debugChannels=[], append_to_last_log=False, live_plot=False):

        if log_directory=="":
            dirname = os.path.dirname(__file__)
            log_directory = dirname+"/Models/"+name

        self.live_plot = live_plot
        self.plot, self.subplots = None, []
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
                #print(filename)
                if file_extension==".log":
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
                    #increment last log number and turn it to string

                    if append_to_last_log:
                        last_log = "."+str(int(last_log[len(last_log)-2]))
                    else:
                        last_log = "."+str(int(last_log[len(last_log)-2])+1)

                #In this case the last log file with same name was the first one
                else:
                    if append_to_last_log:
                        last_log = ".0"
                    else:
                        last_log = ".1"

            if last_log=="":
                last_log=".0"
                append_to_last_log=False

            self.log_filename = log_directory + "/" + self.name + last_log + ".log"
            self.history_log_filename = log_directory + "/" + self.name + last_log + ".hst"
        else:
            self.log_filename = log_filename
            self.history_log_filename = history_log_filename

        #The generic channel is the default one
        self.debugChannels = ["Generic"]
        self.debugChannels.append("Logger")

        for dl in debugChannels:
            self.debugChannels.append(dl)
        
        if append_to_last_log:
            with open(self.log_filename,"a+") as log_file:
                log_file.close()
            #Write to file logger initializations
            with open(self.history_log_filename,"a+") as log_file:
                log_file.close()
        else:
            #Write to file logger initializations
            with open(self.log_filename,"w+") as log_file:
                now = datetime.datetime.now()
                log_file.write("LOGGER started at "+now.strftime("%Y-%m-%d %H:%M:%S.%f")+".\n")
                log_file.write("Currently active debug channels:\n")
                for dc in debugChannels:
                    log_file.write("\t"+dc+"\n")
                log_file.close()
            #Write to file logger initializations
            with open(self.history_log_filename,"w+") as log_file:
                log_file.close()
    
        self.history = {}


    def add_debug_channel(self, name):
        self.debugChannels.append(name)        
        self.log("Debug channel added:\t"+name+"\n",debug_channel="Logger")

    
    def get_debug_channels(self):
        return self.debugChannels
    
    @classmethod
    def caller_name(self, skip=3):
        """Get a name of a caller in the format module.class.method
        
        `skip` specifies how many levels of stack to skip while getting caller
        name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
        
        An empty string is returned if skipped levels exceed stack height
        """
        stack = inspect.stack()
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parentframe = stack[start][0]    
        
        name = []
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        if module:
            name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append( codename ) # function or a method
        del parentframe
        return ".".join(name)
    
    def print(self, *strings, writeToFile = True, debug_channel = "Generic", skip_stack_levels=2):
        log_string=""
        if debug_channel in self.debugChannels:
            now = "["+Logger.getFullDateTime()+"]"
            log_string = now +"["+self.caller_name(skip=skip_stack_levels)+"]["+debug_channel+"]: "
            strings = strings[0]
            for s in strings:
                log_string+=str(s)
            if writeToFile:
                with open(self.log_filename,"a+") as log_file:
                    log_file.write(log_string+"\n\n")
                    log_file.close()
            print(log_string)

    def log(self, *strings, debug_channel = "Generic", skip_stack_levels=2):
        if debug_channel in self.debugChannels:
            now = "["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+"]"
            log_string = now +"["+self.caller_name(skip=skip_stack_levels)+"]["+debug_channel+"]: "
            for s in strings:
                log_string+=str(s)
            with open(self.log_filename,"a+") as log_file:
                log_file.write(log_string+"\n\n")
                log_file.close()
    
    
    def log_history(self, **values):
        history_values = values
        log_string=""

        #print("AAAAAAAAAAAA",self.history)
        #exit()

        for history_key,history_value in history_values.items():
            if history_key in self.history:
                self.history[history_key].append(history_value)
                #print("11111111111111: ",history_key)
            else:
                #print("222222222222",history_key)
                self.history[history_key] = []
                self.history[history_key].append(history_value)
        
        #print(self.history)

        for k,v in self.history.items():
            log_string+=""+str(k)+":"+str(v)+";"

        #print(log_string)

        with open(self.history_log_filename,"w+") as log_file:
            log_file.write(log_string)
            log_file.close()

        if self.live_plot:
            if self.plot==None:
                #print(history_values.keys())
                self.plot, self.subplots = plt.subplots(len(history_values.keys()), 1, constrained_layout=True)
            self.live_plot_history()
    
    def load_history(self, filename="", to_episode=-1):
        if filename=="": filename = self.history_log_filename
        if not filename.endswith(".hst"): filename+=".hst"
        self.history = {}
        with open(filename, "r") as history_file:
            firstline = history_file.readline()
            #in the first version of this history logger the info was stored incrementally
            #so all the info is contained in the last line
            #print(firstline)
            if firstline=="v1\n":
                all_lines = history_file.readlines()
                lastline=all_lines[len(all_lines)-1]
                if lastline=="": return
                dict_contents = lastline.split("{")[1].split("}")[0]
                dict_contents = dict_contents.split(":")
                result=[]
                for l in dict_contents:
                    result.append(l.replace("'","").replace(" ","").replace("[","").replace("]",""))
                dict_keys = [result[0]]
                for l in result[1:len(result)-1]:
                    l = l.split(",")
                    dict_keys.append(l[len(l)-1])
                values = []
                for key in dict_keys:
                    for l in result[1:]:
                        if l.find(key)!=-1:
                            values.append(l.replace(","+key,""))
                values.append(result[len(result)-1])
                value_list = []
                for value in values:
                    value_list.append(value.split(","))
                values = []
                for value_entry in value_list:
                    temp = []
                    for value in value_entry:
                        temp.append(float(value))
                    values.append(temp)
            else:
                dict_contents=firstline.split(";")
                dict_keys = []
                dict_values = []
                for c in dict_contents:
                    if len(c)==0: continue
                    dict_entries = c.split(":")
                    #print(dict_entries)
                    dict_keys.append(dict_entries[0])
                    dict_values.append(dict_entries[1].replace("[","").replace("]","").split(","))
                
                values = []
                for i,value in enumerate(dict_values):
                    #print(value)
                    if to_episode!=-1:
                        value=value[:to_episode+1]
                    values.append([float(v) for v in value])
                            
            for i, key in enumerate(dict_keys):
                self.history[key] = values[i]

            #print("LOAD_HISTORY: ",self.history)
                    
    @staticmethod
    def getFullDateTime():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def create_plot_from_history(self, history_values=None,filename="",from_episode=0,to_episode=0):
        
        if history_values==None:
            history_values=self.history

        plot, subplots = plt.subplots(len(history_values.keys()), 1, constrained_layout=True)
   
        if to_episode==0:
            to_episode=len(next(iter(history_values.values())))

        current_plot=0
        for k,v in self.history.items():
            if len(self.subplots)>1: 
                subplot = subplots[current_plot]
            else:
                subplot = subplots 
            subplot.plot(v[from_episode:to_episode])
            maxval=max(v)
            maxval+=maxval/10
            minval=min(v)
            if minval<0:
                minval+=minval/10
            else:
                minval-=minval/10
                subplot.set_ylim(0,maxval)
            if maxval>0:
                maxval+=maxval/10
            else:
                maxval-=maxval/10
                subplot.set_ylim(minval,0)
            subplot.set_title(k)
            subplot.set_xlabel('Episode')
            subplot.set_ylabel(k)
            current_plot+=1
        
        if filename=="":
            filename = self.history_log_filename.replace("hst","png")
        
        plot.suptitle(self.name)
        plot.savefig(filename)

    def live_plot_history(self):

        plot_color = "#1f77b4"

        history_values=self.history
        #print(history_values)
        current_plot=0
        for k,v in self.history.items():
            self.subplots[current_plot].plot(v, color=plot_color)
            maxval=max(v)
            maxval+=maxval/10
            minval=min(v)
            if minval<0:
                minval+=minval/10
            else:
                minval-=minval/10
                self.subplots[current_plot].set_ylim(0,maxval)
            if maxval>0:
                maxval+=maxval/10
            else:
                maxval-=maxval/10
                self.subplots[current_plot].set_ylim(minval,0)
            self.subplots[current_plot].set_title(k)
            self.subplots[current_plot].set_xlabel('Episode')
            self.subplots[current_plot].set_ylabel(k)
            current_plot+=1
                
        self.plot.suptitle(k, fontsize=20)
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    logger = Logger(name="Prova",log_directory="",debugChannels=["Debug 1","Debug 2"],append_to_last_log=False)
    logger.print("HELLO")
    logger.load_history("MountainCar-v0")
    logger.create_plot_from_history(to_episode=300)
