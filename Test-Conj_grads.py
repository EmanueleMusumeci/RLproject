import tensorflow as tf
import numpy as np
from utils import print_debug

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

def conjugate_grad1(Ax, b):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    old_p = p.copy()
    r_dot_old = np.dot(r,r)
    print("r_dot_old: ",r_dot_old)
    for _ in range(self.cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + 1e-8)
        old_x = x
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        beta = r_dot_new / (r_dot_old + 1e-8)
        r_dot_old = r_dot_new
        if r_dot_old < self.residual_tol:
            break
        old_p = p.copy()
        p = r + beta * p
        if np.isnan(x).any():
            print("x is nan")
            print("z", np.isnan(z))
            print("old_x", np.isnan(old_x))
            print("kl_fn", np.isnan(kl_fn()))
    return x

#For testing this will return a matrix A = [[4,1],[1,3]]
def fisher_vectors_product(model,p,damping):
    A = np.matrix("4. 1.; 1. 3.")
    print_debug("A: ",A )
    return A

def hessian_vector_product(p):
    def hvp_fn(): 
        kl_grad_vector = flatgrad(kl_fn, self.model.trainable_variables)
        grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
        return grad_vector_product

    fisher_vector_product = flatgrad(hvp_fn, self.model.trainable_variables).numpy()
    return fisher_vector_product + (self.cg_damping * p)

A = fisher_vectors_product(None,0,0)
b = np.array([1.,2.])
expected_result = np.matrix(" 0.0909; 0.6364")
print("Testing conjugate gradients implementation: \n Input: \nA:\n",A,"\nb:\n",b,"\nResult should be:\n",expected_result)
result = conjugate_gradients(None,b,0,1000)
#result = conjugate_grad1(A,b)
print("Result: ", result)