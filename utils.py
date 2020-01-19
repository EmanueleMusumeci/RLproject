import math
import numpy as np
import inspect

import tensorflow as tf

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


#DONE#TODO: Studiare e rifattorizzare
#    #TODO: Testare se funziona correttamente
def conjugate_gradients(model, policy_gradient, damping_factor, nsteps, old_action_probabilities, new_action_probabilities, residual_tol=1e-10):
    result = np.zeros_like(policy_gradient)
    r = tf.identity(policy_gradient)
    p = tf.identity(policy_gradient)

#    #TODO: verificare se equivalente a torch.dot
    rdotr = np.dot(r, r)

    #This should peform len(policy_gradient) iterations at most
    nsteps = min(nsteps, len(policy_gradient))

    for _ in range(nsteps):
        #Verificare se il risultato di fisher_vectors_product è A o A * p
        temp_fisher_vectors_product = fisher_vectors_product(model, old_action_probabilities, new_action_probabilities, p, damping_factor)
        
        alpha = rdotr / np.dot(p, temp_fisher_vectors_product)

        result += alpha * p
        r -= alpha * temp_fisher_vectors_product
        new_rdotr = np.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return result

def fisher_vectors_product(model, old_policy_probabilities, new_policy_probabilities, step_direction_vector, damping_factor):
    #this method is supposed to compute the Fishers's vector product as the Hessian of the Kullback-Leibler divergence

    args = {
        "old_policy_probabilities" : old_policy_probabilities,
        "new_policy_probabilities" : new_policy_probabilities,
    }

    #grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    kl_divergence_gradients = model.get_flat_gradients(get_mean_kl_divergence, args)
    #kl_flat_gradients = tf.concat([tf.reshape(gradient, [-1]) for gradient in kl_divergence_gradients], axis=0)

    #kl_v = (flat_grad_kl * Variable(v)).sum()
    def kl_value(): 
        return tf.reduce_sum(kl_divergence_gradients * tf.Variable(step_direction_vector))
    #kl_value_grads = torch.autograd.grad(kl_v, model.parameters())

    #This is the actual Hessian value
    kl_flat_grad_grads = model.get_flat_gradients(kl_value)
    
    #flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

    return kl_flat_grad_grads + step_direction_vector * damping_factor

def mean_kl_divergence(old_action_probabilities,action_probabilities):
    return tf.reduce_mean(tf.reduce_sum(old_action_probabilities * tf.math.log(old_action_probabilities / action_probabilities), axis=1))


def get_mean_kl_divergence(args):
    old_policy_probabilities = args["old_policy_probabilities"]
    new_policy_probabilities = args["new_policy_probabilities"]
    return tf.reduce_mean(tf.reduce_sum(old_policy_probabilities * tf.math.log(old_policy_probabilities / new_policy_probabilities), axis=1))


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

