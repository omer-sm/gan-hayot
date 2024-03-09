from ML.DL import *
import sys
from PIL import Image
from ML.make_input import *
import time
import os

def get_time():
    return time.strftime("%d%m%Y%H%M%S", time.localtime(time.time()))

def open_image(path):
    img = np.array(Image.open(path))
    img = ndimage.zoom(img, (64/img.shape[0], 64/img.shape[1], 1))
    return img

def process_dna(gen, dna):
    if (isinstance(dna, str)):
        img = open_image(dna)
        img = np.array([img]).transpose(3,1,2,0)
        for l in range(2, gen.bottleneck_layer):
            img = gen.layers[l].forward_propagation(img, False)
        return img.T[0]
    return make_single_vector(*dna)

def process_gif_dna(dna1, dna2):
    return make_mat_from_vectors(dna1, dna2)


def generate_and_show(is_gif, model_path, dna1, dna2 = None, m = 0, framerate = 0):
    gen = DLModel(use_cuda=True)
    gen.add(DLDeflattenLayer("", (4096*3,), (3,64,64)))
    l = DLConvLayer("", 32, (3,64,64), "relu", "He", 0.0003, (4,4), (3,3), "valid", "rmsprop")
    gen.add(l)
    l = DLConvLayer("", 16, l.get_output_shape(), "leaky_relu", "Xavier", 0.0005, (3,3), (2,2), (1,1), "rmsprop")
    gen.add(l)
    l = DLConvLayer("", 16, l.get_output_shape(), "relu", "He", 0.0003, (3,3), (2,2), "valid", "rmsprop")
    gen.add(l)
    l = DLFlattenLayer("", l.get_output_shape())
    gen.add(l)
    gen.add(DLLayer("", 256, l.get_output_shape(), "leaky_relu", "He", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 256, (256,), "leaky_relu", "He", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 512, (256,), "trim_tanh", "Xavier", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 64, (512,), "vae_bottleneck", "Xavier", 8e-5, "rmsprop", samples_per_dim=10)) 
    gen.add(DLLayer("", 256, (320,), "leaky_relu", "He", 8e-5, "rmsprop"))
    gen.add(DLLayer("", 1024, (256,), "leaky_relu", "He", 8e-5, "rmsprop"))
    gen.add(DLLayer("", 4096*3, (1024,), "trim_sigmoid", "Xavier", 8e-5, "rmsprop"))
    '''gen.add(DLDeflattenLayer("", (4096*3,), (3,64,64)))
    l = DLConvLayer("", 32, (3,64,64), "relu", "He", 0.0003, (5,5), (3,3), "valid", "rmsprop")
    gen.add(l)
    l = DLConvLayer("", 32, l.get_output_shape(), "leaky_relu", "Xavier", 0.0005, (3,3), (2,2), (1,1), "rmsprop")
    gen.add(l)
    l = DLConvLayer("", 16, l.get_output_shape(), "relu", "He", 0.0003, (3,3), (2,2), "valid", "rmsprop")
    gen.add(l)
    l = DLFlattenLayer("", l.get_output_shape())
    gen.add(l)
    gen.add(DLLayer("", 256, l.get_output_shape(), "leaky_relu", "He", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 256, (256,), "leaky_relu", "He", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 512, (256,), "trim_tanh", "Xavier", 8e-5, "rmsprop")) 
    gen.add(DLLayer("", 128, (512,), "vae_bottleneck", "Xavier", 8e-5, "rmsprop", samples_per_dim=5)) 
    gen.add(DLLayer("", 256, (320,), "leaky_relu", "He", 8e-5, "rmsprop"))
    gen.add(DLLayer("", 400, (256,), "leaky_relu", "He", 8e-5, "rmsprop"))
    gen.add(DLLayer("", 4096*3, (400,), "trim_sigmoid", "Xavier", 8e-5, "rmsprop"))'''
    gen.add(DLDeflattenLayer("", (4096*3,), (3,64,64)))
    gen.compile("squared_means_KLD", recon_loss_weight=1.3, KLD_beta=0.8)
    gen.load_weights(model_path+"/generator/")

    dna1 = process_dna(gen, dna1)
    out = np.array([dna1]).T
    if (is_gif):
        dna2 = process_dna(gen, dna2)
        out = make_mat_from_vectors(dna1, dna2, m)

    for l in range(gen.bottleneck_layer, len(gen.layers)):
        out = gen.layers[l].forward_propagation(out, False)
    
    if (is_gif):
        out = out.transpose(3, 1, 2, 0)
        imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in out]
        file_path = fr"{os.path.dirname(os.path.abspath(__file__))}\Outputs\output {get_time()}.gif"
        imgs[0].save(file_path, save_all=True, append_images=imgs[1:], duration=(m/framerate), loop=0)
        os.startfile(file_path)
    else:
        out = out.transpose(3, 1, 2, 0)[0]
        img = Image.fromarray((out * 255).astype(np.uint8))
        img.save(fr"{os.path.dirname(os.path.abspath(__file__))}\Outputs\output {get_time()}.jpg")
        img.show()
    

