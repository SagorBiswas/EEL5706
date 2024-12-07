

import sys
import torch
import torchvision.models as models
import struct
import reedsolo
import matplotlib.pyplot as plt
import numpy as np





################################## Hashing for parity number selection #############################
def hash_to_range(value, min_val=4, max_val=32):
    range_size = max_val - min_val + 1
    # Knuth's multiplicative hashing
    hashed_value = (value * 2654435861) % 2**32 + 11
    return min_val + (hashed_value % range_size)

################################## Hashing End #############################


################################## with float kept as it is ########################################
def float_to_bytes(f_val):     # Convert a float to 4 bytes
    return struct.pack('f', f_val)


def flatten_and_pack_weights(weights):      # Flatten weights and convert them into a bytearray
    flattened_weights = weights.flatten()
    byte_list = bytearray()
    for weight in flattened_weights:
        byte_list.extend(float_to_bytes(weight))
    return byte_list


def generate_polynomial(encoded_data):
    # Polynomial coefficients are the elements of the encoded data
    polynomial = np.poly1d(encoded_data[::-1])  # Reverse the list as the lowest degree term is last
    return polynomial

def generate_rs_parity_v2(weights, p):
    if p <= 0:
        raise ValueError("Number of parity bits (p) must be greater than 0.")
    
    # Convert to Bytearray
    byte_data = flatten_and_pack_weights(weights)

    # Generate RS Parity
    rs = reedsolo.RSCodec(p)  # Initialize Reed-Solomon encoder
    encoded = rs.encode(byte_data)  # rs encode the byte data

    # Extract parity bits (last `p` bytes)
    parity_bits = encoded[-p:]  # Last `p` bytes are parity

    # print_status
    # print("Encoded Data Length:", len(encoded))
    # print("Parity Bits:", list(parity_bits))    
    # return list(parity_bits)
    
    
    # # Create the polynomial representation
    # poly_terms = list(data_values) + list(parity_bits)  # Combine data and parity
    # polynomial = " + ".join(
    #     f"{coef}x^{i}" for i, coef in enumerate(poly_terms[::-1])
    # )  # Polynomial in descending order

    polynomial = generate_polynomial(encoded)
    return list(parity_bits), polynomial

################################## with float kept as it is End ########################################




################################## Server side codes #############################################

class Weight_varifier:
    def __init__(self, parity_bits):
        self.__stored_parity = parity_bits

    def encode(self):
        print(f"Variable 1: {self.var1}")
        print(f"Variable 2: {self.var2}")

    def update_variables(self, new_var1, new_var2):
        self.var1 = new_var1
        self.var2 = new_var2
        print("Variables updated successfully.")

    def generate_rs_parity(self, weights, layer=0, kernel_indx=0):
        p = hash_to_range (layer+kernel_indx)
        
        # Convert to Bytearray
        byte_data = flatten_and_pack_weights(weights)

        # Generate RS Parity
        rs = reedsolo.RSCodec(p)  # Initialize Reed-Solomon encoder
        encoded = rs.encode(byte_data)  # rs encode the byte data

        # Extract parity bits (last p bytes)
        parity_bits = encoded[-p:]  
        
        return list(parity_bits)


    def verify(self):
        try:
            return self.var1 + self.var2
        except TypeError:
            print("Variables are not numeric and cannot be added.")
            return None
        




################################## User side varification #########################################

def user_side_verifier(downloaded_weights, server_parity, layer=0, kernel_indx=0):

    # the system generates different number of parities based on layer and kernel
    p = hash_to_range(layer+kernel_indx) 

    # Generate parity for user's weights
    user_parity = generate_rs_parity_v2(downloaded_weights, p)
    # print(user_parity)
    # Compare parity
    return user_parity[0] == server_parity
    # return Weight_varifier.verify(user_parity)

def user_side_verifier(downloaded_weights, layer=0, kernel_indx=0):

    # the system generates different number of parities based on layer and kernel
    p = hash_to_range(layer+kernel_indx) 

    # Generate parity for user's weights
    user_parity = generate_rs_parity_v2(downloaded_weights, p)

    # make call to server class to Compare parity
    return Weight_varifier.verify(user_parity)



################################## User side varification End #########################################






############################################ Model Extraction ######################################################

def extract_conv1_kernels(model):
    # Get the Conv1 layer from ResNet18
    conv1_layer = model.conv1
    
    # Get the kernel weights of Conv1 (size: [out_channels, in_channels, height, width])
    kernels = conv1_layer.weight.data  # This is a tensor with shape [64, 3, 7, 7]
    
    # Convert the tensor to a list or numpy array, if needed
    kernels_list = kernels.cpu().numpy()  # Moving to CPU and converting to numpy (if on GPU)
    
    return kernels_list


#################################################### Model extraction End ###########################################




######################################### generate reed-solomon #####################################################

def convert_data_to_byte(data_list):
    min_val = np.min(data_list)
    max_val = np.max(data_list)
    normalized_data_list = (data_list - min_val) / (max_val - min_val)
    scaled_data_list = normalized_data_list * 255
    mapped_data_list = scaled_data_list.astype(int)

    return mapped_data_list


def draw_graph(X_axis, Y_axis, x_label='weight', y_label = 'parity'):

    # Now plot the graph
    plt.plot(X_axis, Y_axis , label='weight/parity value', marker='o', linestyle='-')  # , color='b')
        
    # plt.ylim(-0.05, +0.05)  # Adjust the range to span 20 (from 80 to 100)

    plt.xticks(range(0, len(x_axis)+1, 4))  # Adjust tick frequency to every 10 units
    # plt.yticks(range(0, 257, 32))  # Adjust tick frequency to every 10 units

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} VS weight/parity value")  #  {y_label}")
    # Show legend
    plt.legend()
    # Show the plot
    plt.show()

######################################### generate reed-solomon end #################################################



if __name__ == '__main__':

    
    # Load pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    output_no = 0
    input_chan = 0
    layer = 0
    kernel_indx = input_chan*3 + output_no

    # Extract Conv1 kernels
    conv1_kernels = extract_conv1_kernels(model)

    # Display the shape and some of the extracted kernels
    print("Shape of Conv1 kernels:", conv1_kernels.shape)
    print("Shape of Conv1 first kernel:", conv1_kernels[input_chan][output_no].shape)
    print("First kernel in Conv1 layer:\n", conv1_kernels[input_chan][output_no])  # Display the first kernel
    print ("#"*150)

    # store the actual weight's parity
    p = hash_to_range(layer + kernel_indx)
    server_parity_bits, poly = generate_rs_parity_v2(conv1_kernels[input_chan][output_no], p)
    print("\nParity bits=>", server_parity_bits)

    print("*"*100)
    print(poly) 
    print("*"*100)


    # data = [round(elm, 5) for elm in conv1_kernels[0][0].flatten()]
    data = conv1_kernels[input_chan][output_no].copy().flatten()
    # print("Data => ", data)
    data_list = convert_data_to_byte(data.copy()).tolist()
    data_list.extend(server_parity_bits)
    x_axis = [elm for elm in range(len(data_list))]
    draw_graph(x_axis, data_list, 'data position', 'weight/parity data value in finite field')


    ################ user side ################
    user_model = models.resnet18(pretrained=True)
    user_conv1_kernels = extract_conv1_kernels(model)
    downloaded_weights = user_conv1_kernels[input_chan][output_no]  #.copy()  # Assume user downloads unmodified weights
    # print("Downloaded model. shape = ", downloaded_weights.shape)
    is_valid = user_side_verifier(downloaded_weights, server_parity_bits, layer, kernel_indx)    
    
    # print("Downloaded weights =>", downloaded_weights)

    print("$"*100)

    if is_valid:
        print("Weights are valid and unmodified.")
    else:
        print("Weights have been tampered with!")



############################################ Saved Outputs #########################################################
