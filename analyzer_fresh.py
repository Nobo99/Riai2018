import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res
   
def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon
     
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0


def linear_solver(inf, sup, weights, biases, numlayers,output,start,label):

    opt_model = Model("mip1")
    opt_model.setParam('OutputFlag', False)

    for layer in range(start,numlayers):

       
        if (layer+1!=nn.numlayer):
            layer_weights = nn.weights[layer+1]
            layer_biases = nn.biases[layer+1]
            np.ascontiguousarray(layer_weights, dtype=np.double)
            np.ascontiguousarray(layer_biases, dtype=np.double)
            num_out_pixels = len(layer_biases)
            num_in_pixels = len(layer_weights[0])
            
        else:
            num_out_pixels = 10
            num_in_pixels = 10

        set_I = range(num_in_pixels) 
        set_J = range(num_out_pixels)
        set_optimize_I = [i for i in set_I if inf[layer][i]<0 and sup[layer][i]>0]
        set_zeros_I = [i for i in set_I if inf[layer][i]<=0 and sup[layer][i]<=0]
        set_leave_I = [i for i in set_I if inf[layer][i]>0 and sup[layer][i]>0]
        lambda_ = [0]*num_in_pixels
        mu = [0] * num_in_pixels

        for i in set_optimize_I:
            lambda_[i]=(sup[layer][i]/(sup[layer][i]-inf[layer][i]))

        for i in set_optimize_I:
            mu[i]=((-1)*(sup[layer][i]*inf[layer][i])/(sup[layer][i]-inf[layer][i]))
        

        h_i_vars  = opt_model.addVars(num_in_pixels, lb=inf[layer], ub=sup[layer], name='h{0},'.format(layer))

        if(layer!=start):
            constraints_eq = {i: opt_model.addConstr(lhs=objective[i],sense=GRB.EQUAL,rhs=h_i_vars[i],
                                                 name="prev_constraint_{0}".format(i)) for i in set_I}
        
        relu_h_i_vars  = opt_model.addVars(num_in_pixels, name='relu_h{0},'.format(layer))

        if (layer+1!=numlayers):
            objective = [quicksum(layer_weights[j][i] * relu_h_i_vars[i] for i in set_I) + layer_biases[j]
                         for j in set_J]
        else: 
            objective = [relu_h_i_vars[i]*1 for i in set_I]

        constraint_lq = {i : opt_model.addConstr(lhs=relu_h_i_vars[i],sense=GRB.LESS_EQUAL,
                                               rhs=lambda_[i] * h_i_vars[i] + mu[i], 
                                               name="constraint_lq_{0},{1}".format(layer,i))
                       for i in set_optimize_I}

        constraint_zero = {i : opt_model.addConstr(lhs=relu_h_i_vars[i],sense=GRB.GREATER_EQUAL,rhs=0,
                                               name="constraint_zero_{0},{1}".format(layer,i))
                       for i in set_optimize_I}

        constraints_grq = {i :opt_model.addConstr(lhs=relu_h_i_vars[i],sense=GRB.GREATER_EQUAL,rhs=h_i_vars[i],
                                              name="constraint_grq_{0},{1}".format(layer,i)) for i in set_optimize_I}

        constraints_positive = {i : opt_model.addConstr(lhs=relu_h_i_vars[i],sense=GRB.EQUAL,rhs=h_i_vars[i],
                                               name="constraint_{0},{1}".format(layer,i)) 
                       for i in set_leave_I}

        constraints_negative = {i : opt_model.addConstr(lhs=relu_h_i_vars[i],sense=GRB.EQUAL,rhs=0,
                                               name="constraint_{0},{1}".format(layer,i)) 
                      for i in set_zeros_I}
        
        

    if (numlayers != nn.numlayer):

        opt_model.setObjective(objective[output], GRB.MINIMIZE)
        opt_model.optimize()
        h_j_min = opt_model.objVal
        opt_model.setObjective(objective[output],GRB.MAXIMIZE)
        opt_model.optimize()
        h_j_max = opt_model.objVal
        return h_j_min, h_j_max
    
    
    else:
        opt_model.setObjective(objective[label]-objective[output],GRB.MINIMIZE)
        opt_model.setParam('OutputFlag', False)
        opt_model.optimize()
        value = opt_model.objVal
        return value , 0
        

def analyze(nn, LB_N0, UB_N0, label, netname, classify=False):
    
    start = 0
    myinf = []
    mysup = []
    not_all_neurons = False
    final_layer = nn.numlayer
    not_verified_flag = False
    
    
    if (netname.endswith("mnist_relu_3_10.txt") or netname.endswith("mnist_relu_3_20.txt") or 
        netname.endswith("mnist_relu_3_50.txt")):
        stop = 2
    if (netname.endswith("mnist_relu_6_20.txt") or netname.endswith("mnist_relu_6_50.txt")):
        stop = 2
    if (netname.endswith("mnist_relu_6_100.txt")):
        stop = 3
        not_all_neurons = True
        neuron_ratio = 0.6
    if (netname.endswith("mnist_relu_9_100.txt")):
        stop = 3
        not_all_neurons = True
        neuron_ratio = 0.5
    if (netname.endswith("mnist_relu_6_200.txt")):
        stop = 3
        not_all_neurons = True
        neuron_ratio = 0.3
    if (netname.endswith("mnist_relu_9_200.txt")):
        stop = 3
        not_all_neurons = True
        neuron_ratio = 0.25
    if (netname.endswith("mnist_relu_4_1024.txt")):
        not_all_neurons = True
        stop = 2
        neuron_ratio = 0.1
        final_layer = 2
    stepsize = stop
    
        
        
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)

           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           # handle affine layer
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)

           bounds = elina_abstract0_to_box(man, element)
           sup = [bounds[i].contents.sup.contents.val.dbl for i in range(num_out_pixels)]
           inf = [bounds[i].contents.inf.contents.val.dbl for i in range(num_out_pixels)]
           myinf.append(inf)
           mysup.append(sup)

           # handle ReLU layer 
           if(nn.layertypes[layerno]=='ReLU'):
              element = relu_box_layerwise(man,True,element,0, num_out_pixels)
           nn.ffn_counter+=1 
        
           if(layerno+1>=stop and layerno+1 <= final_layer and not classify):
               rnd = range(0,num_out_pixels)


               if (not stop == nn.numlayer and not_all_neurons):
                    rnd = np.random.permutation(num_out_pixels)
                    index = round(num_out_pixels*neuron_ratio)
                    rnd = rnd[0:index]


               for i in rnd:

                    if (stop == nn.numlayer):
                        low, high = linear_solver(myinf,mysup,nn.weights,nn.biases,stop,i,start,label)
                        if (low<0):
                            not_verified_flag = True
                            break

                    else:
                        low, high = linear_solver(myinf,mysup,nn.weights,nn.biases,stop,i,start,label)

                        #create an array of two linear constraints
                        lincons0_array = elina_lincons0_array_make(2)
                        #Create a greater than or equal to inequality for the lower bound
                        lincons0_array.p[0].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                        linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                        cst = pointer(linexpr0.contents.cst)
                        #plug the lower bound “a” here
                        elina_scalar_set_double(cst.contents.val.scalar, -low)
                        linterm = pointer(linexpr0.contents.p.linterm[0])
                        #plug the dimension “i” here
                        linterm.contents.dim = ElinaDim(i)
                        coeff = pointer(linterm.contents.coeff)
                        elina_scalar_set_double(coeff.contents.val.scalar, 1)
                        lincons0_array.p[0].linexpr0 = linexpr0
                        #create a greater than or equal to inequality for the upper bound
                        lincons0_array.p[1].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                        linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                        cst = pointer(linexpr0.contents.cst)
                        #plug the upper bound “b” here
                        elina_scalar_set_double(cst.contents.val.scalar, high)
                        linterm = pointer(linexpr0.contents.p.linterm[0])
                        #plug the dimension “i” here
                        linterm.contents.dim = ElinaDim(i)
                        coeff = pointer(linterm.contents.coeff)
                        elina_scalar_set_double(coeff.contents.val.scalar, -1)
                        lincons0_array.p[1].linexpr0 = linexpr0
                        #perform the intersection
                        element = elina_abstract0_meet_lincons_array(man,True,element,lincons0_array)
               if (stop>=stepsize):
                    start +=1

               stop +=1

           print("Layer {} finished".format(layerno))
        else:
           print(' net type not supported')
   
    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)

           
    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break    
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, (verified_flag or (not not_verified_flag))



if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    classify = True
    label, _ = analyze(nn,LB_N0,UB_N0,0,netname,classify)
    start = time.time()
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label,netname)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
    

