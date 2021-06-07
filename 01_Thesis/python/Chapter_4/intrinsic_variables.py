import numpy as np
import matplotlib.pyplot as plt



#load the full order BGK data
def load_BGKandMethod():
    if (method == 'Fully' or method=="POD") and level == 'hy':
        c = np.load('Preprocessed_Data/sod25Kn0p00001_2D_unshuffled.npy')
    elif (method == 'Fully' or method=="POD") and level == 'rare':
        c = np.load('Preprocessed_Data/sod25Kn0p01_2D_unshuffled.npy')
    elif method == 'Conv' and level == 'hy':
        c = np.load('Preprocessed_Data/sod25Kn0p00001_4D_unshuffled.npy')
    elif method == 'Conv' and level == 'rare':
        c = np.load('Preprocessed_Data/sod25Kn0p01_4D_unshuffled.npy')   

    print("Method:",method,"Level:",level)
    return c


#Plot the results of the variation of intrinsic variables
fig,axs = plt.subplots(1,2)
i = 0
for idx, level in enumerate(["hy", "rare"]):
    pod = []
    fully = []
    conv = []
    for iv in [1,2,4,8,16,32]:
        print(iv)
        
        #For POD
        method = "POD"
        c = load_BGKandMethod() # load FOM data for evaluation
        from POD import intr_eval
        l2_pod = intr_eval(c,iv)
        pod.append(l2_pod)

        #For Fully
        method = "Fully"
        c = load_BGKandMethod() # load FOM data for evaluation
        from FullyConnected import intr_eval
        l2_fully = intr_eval(c,iv,level)
        fully.append(l2_fully)
        

        #For Conv
        method = "Conv"
        c = load_BGKandMethod() # load FOM data for evaluation
        from Conv import intr_eval
        l2_conv = intr_eval(c,iv,level)
        conv.append(l2_conv)
        continue

    axs[i].semilogy([1,2,4,8,16,32],pod,'k''x',label="POD")
    #axs[i].semilogy([1,2,4,8,16,32],fully,'r''o',label="Fully")
    #axs[i].semilogy([1,2,4,8,16,32],conv,'g''v',label="Conv")
    axs[i].grid(True,which="both")
    print(level)
    axs[i].legend()
    i+=1
#tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/01_Thesis/Figures/Results/Var_iv.tex')
plt.show()