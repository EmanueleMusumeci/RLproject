from collections import namedtuple
import os
from os.path import abspath
import glob
import logging
import inspect
import errno


import datetime

import matplotlib.pyplot as plt

HistoryValues = namedtuple("HistoryValues","max_reward kl_div value_loss policy_loss")

class Logger:
    def __init__(self, name=None, log_directory="TRPO_project/Models", log_filename="", history_log_filename="", debugChannels=[]):

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
                    print(last_log)
                    #increment last log number and turn it to string
                    last_log = "."+str(int(last_log[len(last_log)-2])+1)
                    print(last_log)
                #In this case the last log file with same name was the first one
                else:
                    last_log = ".1"
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
        
        #Write to file logger initializations
        with open(self.log_filename,"w+") as log_file:
            now = datetime.datetime.now()
            log_file.write("LOGGER started at "+now.strftime("%Y-%m-%d %H:%M:%S.%f")+".\n")
            log_file.write("Currently active debug channels:\n")
            for dc in debugChannels:
                log_file.write("\t"+dc+"\n")
            log_file.close()
    
        self.history = []

        #Write to file logger initializations
        with open(self.history_log_filename,"w+") as log_file:
            log_file.close()

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
    
    
    def log_history(self, max_reward, kl_div, value_loss, policy_loss):
        history_values = HistoryValues(max_reward,kl_div,value_loss,policy_loss)
        self.history.append(history_values)

        log_string=str(max_reward)
        log_string+=","+str(kl_div)
        log_string+=","+str(value_loss)
        log_string+=","+str(policy_loss)
        with open(self.history_log_filename,"a+") as log_file:
            log_file.write(log_string+"\n")
            log_file.close()
    
    def load_history(self, filename=""):
        if filename=="": filename = self.history_log_filename
        self.history = []
        with open(filename, "r") as history_file:
            lines = history_file.readlines()
            for line in lines:
                line = line.split(",")
                assert(len(line)==5), "Malformed history"
                self.history.append(HistoryValues(line[0],line[1],line[2],line[3],line[4]))

    def plot_history(self):
        policy_losses = []
        value_losses = []
        for hv in self.history:
            policy_losses.append(hv.policy_loss)
            value_losses.append(hv.value_loss)
        plt.plot(policy_losses, "orange")
        plt.plot(value_losses, "blue")
        plt.show()
    
    @staticmethod
    def getFullDateTime():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

if __name__ == "__main__":
    logger = Logger(name="Prova",debugChannels=["Debug 1","Debug 2"])
    logger.log("ciao", ", come va? Sono un logger", debug_channel="Debug 1")
    logger.log("ciao", ", come va? Sono un logger", debug_channel="Debug 2")
    logger.print("ciao",", Questo dovrebbe apparire solo sul terminale", debug_channel="Debug 1")
    logger.print("ciao",", Questo dovrebbe apparire solo sul terminale", debug_channel="Debug 2")
    logger.print("ciao",", Questo dovrebbe apparire sia sul terminale che nel log", writeToFile=True,debug_channel="Debug 1")
    logger.print("ciao",", Questo dovrebbe apparire sia sul terminale che nel log", writeToFile=True, debug_channel="Debug 2")
    logger.print("Generic channel","!!!")