import tensorflow as tf
import numpy as np
from utils import print_debug

EPS = 1e-8

def conjugate_gradients(model, policy_gradient, damping_factor, nsteps, residual_tol=1e-10):
    result = np.zeros_like(policy_gradient)
    #r = policy_gradient.clone()
    #p = policy_gradient.clone()

    r = policy_gradient.copy()
    print("r: \n",r)
    p = policy_gradient.copy()
    print("p: \n", p)
#    #TODO: verificare se equivalente a torch.dot
    rdotr = np.dot(r, r)
    print("rdotr: ",rdotr)

    #This should peform len(policy_gradient) iterations at most
    nsteps = min(nsteps, len(policy_gradient))

    for i in range(nsteps):
        #Verificare se il risultato di fisher_vectors_product è A o A * p
        temp_fisher_vectors_product = fisher_vectors_product(model, p, damping_factor)
        
        Ap = np.dot(p, temp_fisher_vectors_product)
        print("rdotr: ",rdotr,", Ap: ",Ap)
        alpha = rdotr / Ap
        
        
        result += alpha * p
        r -= alpha * temp_fisher_vectors_product
        new_rdotr = np.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return result

def conjugate_gradients(self, model, policy_gradient, damping_factor, cg_iters, args, residual_tol=1e-10, logger=None):
        result = np.zeros_like(policy_gradient)
        r = tf.identity(policy_gradient)
        p = tf.identity(policy_gradient)

        rdotr = np.dot(r, r)

        for i in range(cg_iters):
            if logger!=None:
                log("Conjugate gradients iteration: ",i, writeToFile=True,debug_channel="utils", skip_stack_levels=3, logger=logger)

            #Verificare se il risultato di fisher_vectors_product è A o A * p
            temp_fisher_vectors_product = self.fisher_vectors_product(model, p, damping_factor, args, logger)
            
            alpha = rdotr / (np.dot(p, temp_fisher_vectors_product) + EPS)

            result += alpha * p
            r -= alpha * temp_fisher_vectors_product
            new_rdotr = np.dot(r, r)
            beta = new_rdotr / rdotr + EPS
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return result

#For testing this will return a matrix A = [[4,1],[1,3]]
def fisher_vectors_product(model, step_direction_vector, damping_factor, args, logger=None):
    #this method is supposed to compute the Fishers's vector product as the Hessian of the Kullback-Leibler divergence

    def get_mean_kl_divergence(args, logger=None):
        current_batch_observations = args["observation"]
        current_batch_actions_one_hot = args["actions_one_hot"]            

        #4.5) Compute new policy using latest policy (update with latest batch) and old policy (the one that dates back to the latest rollout) 
        #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
        new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
        new_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * new_policy_action_probabilities, axis=1)
        
        #Calcola la vecchia policy usando la vecchia rete neurale salvata
        old_policy_action_probabilities = tf.nn.softmax(self.policy(current_batch_observations, tensor=True))
        old_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * old_policy_action_probabilities, axis=1)

        kl = tf.reduce_mean(old_policy_action_probabilities * tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities))
        if math.isinf(kl):
            if logger!=None:
                log("Inf divergence: old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
        elif math.isnan(kl):
            if logger!=None:
                log("nan divergence: old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
        else:
            if logger!=None:
                log("Divergence: ",kl,", old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
        return kl


    def kl_value(logger=None):
        kl_divergence_gradients = model.get_flat_gradients(get_mean_kl_divergence, args)
        if logger!=None:
            log("KL divergence gradients: ",kl_divergence_gradients,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)

        kl = tf.reduce_sum(kl_divergence_gradients * step_direction_vector)
        return kl

        #This is the actual Hessian value
        kl_flat_grad_grads = model.get_flat_gradients(kl_value)

        if logger!=None:
            log("KL divergence Hessian: ",kl_flat_grad_grads,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)

        return kl_flat_grad_grads + step_direction_vector * damping_factor

A = fisher_vectors_product()
b = np.array([1.,2.])
expected_result = np.matrix(" 0.0909; 0.6364")
print("Testing conjugate gradients implementation: \n Input: \nA:\n",A,"\nb:\n",b,"\nResult should be:\n",expected_result)
result = conjugate_gradients(None,b,0,1000)
#result = conjugate_grad1(A,b)
print("Result: ", result)