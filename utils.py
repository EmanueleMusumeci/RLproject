import math
import numpy as np
import inspect

import tensorflow as tf
from Logger import Logger

EPS = 1e-8

#DONE#TODO: Studiare e rifattorizzare


def mean_kl_divergence(old_policy_probabilities,new_policy_probabilities, logger=None):
    kl = tf.reduce_mean(tf.reduce_sum(old_policy_probabilities * tf.math.log(old_policy_probabilities / new_policy_probabilities),axis=1))
    if math.isinf(kl):
        if logger!=None:
            log("Inf divergence: old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    elif math.isnan(kl):
        if logger!=None:
            log("nan divergence: old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    else:
        if logger!=None:
            log("Divergence: ",kl,", old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    return kl


def caller_name(skip=2):
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

def print_debug(obj=None,debug=True,*strings):
    if not debug: return
    if obj!=None and hasattr(obj,"debug"):
        if obj.debug: 
            print("[",caller_name(),"] ", end="")
            for s in strings:
                print(str(s), end="")
            print()
    else:
        print("[",caller_name(),"]: ", end="")
        for s in strings:
            print(str(s), end="")
        print()

def log(*strings, writeToFile=False, debug_channel="Generic", skip_stack_levels=2, logger=None):
    strings = [str(s) for s in strings]
    if logger!=None:
        logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel, skip_stack_levels=skip_stack_levels)
