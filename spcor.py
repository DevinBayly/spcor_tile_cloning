from pathlib import Path
import scipy
import skimage
from skimage.restoration import inpaint
import hashlib

import matplotlib.pyplot as plt

import pyvips

import numpy as np

from PIL import Image

import imagehash
Image.MAX_IMAGE_PIXELS=None

size = 15
half = size//2
double=size*2
import scipy


def normalized_to_uint8(im):
    return (im*255).round().astype("uint8")


def tile_motivated_normalized_search(usgs_tile,drone,size):
    # match the histogram to the ref
    norm_usgs = normalized_to_uint8(skimage.exposure.equalize_hist(np.array(usgs_tile)))
    norm_usgs_im = Image.fromarray(norm_usgs)
    # now go over the drone tiles and equalize each one
    # shouldn't worry about whether the final pick is wrong, because we can repeat with a smaller set of matched histogram items?
    # now construct the tree and map the way we did previously
    tree,map = create_query_structures(drone,size,random_number=30000,equalize=True)
    # get neibs = 
    neibs = return_mapped_neighbors(norm_usgs_im,tree,map,32)
    # attempt a best retrieval
    best = pick_perceptually_best_neighbor(norm_usgs_im,neibs,hash_size=8,rotation=45)
    return best,neibs

def tile_motivated_search(usgs_tile,drone,size):
    # match the histogram to the ref
    usgs_im = Image.fromarray(usgs_tile)
    # now go over the drone tiles and equalize each one
    # shouldn't worry about whether the final pick is wrong, because we can repeat with a smaller set of matched histogram items?
    # now construct the tree and map the way we did previously
    tree,map = create_query_structures(drone,size,random_number=50000)
    # get neibs = 
    neibs = return_mapped_neighbors(usgs_im,tree,map,64)
    # attempt a best retrieval
    best = pick_perceptually_best_neighbor(usgs_im,neibs,rotation=90)
    return best,neibs


def hash_3d_array(arr_3d):
    bytes = arr_3d.tobytes()
    return hashlib.sha256(bytes).hexdigest()

def random_tile_up(im,size,rand_counts):
    ## assumption is tha twe are working with square images
    rand_coords =np.unique((np.random.random((rand_counts,2))*im.width).astype("int64"),axis=0)
    tiles = []
    for rand in rand_coords:
        tiles.append(im.crop((rand[1],rand[0],rand[1]+size,rand[0]+size)))
    return tiles


# consider that we will probably need to be aware that the size will change everything about the lookup that we can perform, more pixls in tile equals larger 3d point 
# function that will combine a few different steps here, we take in a large image, and a size of tile, and then what we produce is a kdtree,  and map
# this will enable us to later on make a new 3d point 
def create_query_structures(im,size,random_number=0,equalize=False):
    if random_number ==0:
        tiles = tile_up_image(im,size)
    else:
        tiles = random_tile_up(im,size,random_number)
    equalized_drone_tiles = []
    if equalize:
        for rand in tiles:
            rand_arr = np.array(rand)
            eq_rand = normalized_to_uint8(skimage.exposure.equalize_hist(rand_arr))
            equalized_drone_tiles.append(eq_rand)
        tiles = equalized_drone_tiles

    map ={}
    arr_3d = [] 
    for i,t in enumerate(tiles):
        el_3d = add_up_channels(t)
        hash = hash_3d_array(el_3d)
        map[hash] = t
        arr_3d.append(el_3d)
    arr_3d = np.array(arr_3d)
    tree = scipy.spatial.KDTree(arr_3d)
    return tree,map

def return_mapped_neighbors(q_im_tile,tree,map,n_neibs):
    im_arr = np.array(q_im_tile)
    q_arr = add_up_channels(im_arr)
    dd,ii = tree.query(q_arr,n_neibs)
    # print("distances",dd)
    hits = tree.data[ii]
    ims=[]
    for h in hits:
        # print(h)
        hash = hash_3d_array(h.astype("int64"))
        ims.append(map[hash])
    return ims

def phash_arr(arr):
    return imagehash.phash(Image.fromarray(arr))


def pick_perceptually_best_neighbor(tile,neighbor_images,do_hash=False,hash_size=8,rotation=90):
    # now we figure out which one has the highest perceptual similarity
    if isinstance(tile,np.ndarray):
        tile = Image.fromarray(tile)
    spec_tile = imagehash.phash(tile,hash_size=hash_size)
    # from the initial neighbors, make a bunch of variations,

    #??? how can we decide which rotations were the best?, should we make a collection of angles that were applied so that we can tell which was the original image that worked?

    others = []
    for neighbor in neighbor_images:
        # do a symmetric flip
        if isinstance(neighbor,np.ndarray):
            neighbor = Image.fromarray(neighbor)
        for i in range(2):
            if i == 1:
                rotation_base = neighbor.transpose(Image.FLIP_LEFT_RIGHT) 
            else:
                rotation_base = neighbor
                # create rotation entries
            for angle in range(0,360,rotation):
                t = rotation_base.rotate(angle)
                # attempt to extend the border pixels 
                if angle%90 !=0:
                    t_arr = np.array(t)
                    tmask = t_arr.sum(axis=2) ==0
                    t_filled = inpaint.inpaint_biharmonic(t_arr,tmask,channel_axis=-1)
                    t = Image.fromarray((t_filled*256).astype("uint8"))
                # seems too flexible in picking wrong images
                if do_hash:
                    hash =imagehash.phash(t,hash_size=hash_size)
                    others.append({"hash":hash,"im":t,"angle":angle,"flip":i==1})
                else:
                    others.append({"hash":False,"im":t,"angle":angle,"flip":i==1})

    print("versions",len(others))
    if do_hash:
        hamming_dist = [spec_tile - e["hash"]  for e in others]
        min_ind_hamming = hamming_dist.index(min(hamming_dist))
    # going to try the normalized version of the coeff
    pearson_coefs = [skimage.measure.pearson_corr_coeff(skimage.exposure.equalize_hist(np.array(tile)),skimage.exposure.equalize_hist(np.array(e["im"]))) for e in others]
    min_ind_pearson = pearson_coefs.index(max(pearson_coefs))
    # print("hamming",min_ind_hamming,"pearson",min_ind_pearson)
    return others[min_ind_pearson]["im"]

def rotate_sources(ims,angle):
    rotated_results = []
    for im in ims:
        rotated_results.append(im.rotate(angle))
    return rotated_results

def make_3d_tile_array(tiles):
    return np.array([add_up_channels(t) for t in tiles])
def add_up_channels(tile):
    tile_arr = np.array(tile)
    channel_sum =tile_arr.sum(axis=0).sum(axis=0)
    return channel_sum
def tile_up_image(pil_image,size):
    tiles = []
    for row in range(0,pil_image.height,size):
        for col in range(0,pil_image.width,size):
            tiles.append(pil_image.crop((col,row,col+size,row+size)))
    return tiles

def make_grid_image(images,size):
    dim = int(len(images)**.5 +1)
    if dim%2 ==1:
        dim +=1
    outer_im = Image.new("RGB",(dim*size,dim*size))
    for i,im_arr in enumerate(images):
        x = i%dim
        y = i//dim
        if isinstance(im_arr,Image.Image): 
            im = im_arr
        else:
            im = Image.fromarray(im_arr)
        outer_im.paste(im,(x*size,y*size))

    return outer_im

def hash_dict_graph(hash_dict,size):
    # make a separate image for each set
    for i,k in enumerate(hash_dict):
        plt.figure()
        images = [e["im"] for e in hash_dict[k]]
        grid_im = make_grid_image(images,size)
        plt.imshow(grid_im)
        



def mk_grid(images,title):
    dim = int(len(images)**.5)
    print(dim)
    fig,ax = plt.subplots(dim,dim)
    for i in range(dim*dim):
        x = i%dim
        y = i//dim
        ax[x,y].imshow(images[i])
    plt.title(title)
    



def make_phash(sub,hs):
    return imagehash.phash(sub,hash_size=hs)

def make_chash(sub,bits):
    return imagehash.colorhash(sub,binbits=bits)

def perform_full_image_tile_hashing(im,size,hs):
    res_map ={}
    for row in range(0,im.height,size):
        for col in range(0,im.width,size):
            part = im.crop((col,row,col+size,row+size))
            hash = make_chash(part,hs)
            string_hash = str(hash)
            res_map[string_hash] = res_map.get(string_hash,[]) + [dict(part=part,converted=hash,row=row,col=col)]
    return res_map

def save_lut_hash_position(base_name,size,bits,lut):
    # strip out the part and converted 
    out_lut = {}
    for hash in lut:
        collect = []
        for e in lut[hash]:
            collect.append(dict(row=e["row"],col = e["col"]))
        out_lut[hash] = collect
    Path(f"{base_name}_{size}_{bits}.json").write_text(json.dumps(out_lut))

import skimage
# TODO consider making a buffer out from the non black pixels so that we especially don't patch fill anywhere that's too close to real content

def pack_single_channel(im):
    im = np.array(Image.fromarray(im.numpy()).convert("RGB"))
    g = im[...,1]
    b = im[...,2]
    r = im[...,0]
    # now convert to shorter ranges of numbers
    # these each get 3 bits
    g = (g/256*8).astype("uint8")
    b = (g/256*8).astype("uint8")
    # 3 this one only two
    r = (g/256*4).astype("uint8")
    # recombine with shifting bits
    g = g<<5
    b = g<<2
    alltogether = r+g+b
    return alltogether
# wonky color
def make_wonky(im):
    cmyk = Image.fromarray(im.numpy()).convert("CMYK")
    trimmed = np.array(cmyk)[:,:,:3]
    hsv_cmy = skimage.color.rgb2hsv(trimmed)
    return hsv_cmy

import matplotlib.patches as patches 



# full = pyvips.Image.new_from_file("full.png")
# part = pyvips.Image.new_from_file("part.png")


def get_zero_positions(part,full,checker=True,skip=5,buffer=True):
    if buffer:
        non_zeros_mask = part[:,:,:3].sum(axis=2) >0
        k = np.ones((skip*2,skip*2))
        expanded_non_zeros  = scipy.signal.convolve2d(non_zeros_mask,k,mode="same")
        # now look for the areas that are false, or not above 0 because this will be the shrunk black pixels zone
        zeros = expanded_non_zeros == False
    else:
        zeros = part[:,:,:3].sum(axis=2) ==0
    coords = np.column_stack(np.where(zeros))
    
      
    selected = np.unique(((coords/skip).round())*skip,axis=0).astype("int")
    # checker pattern the coordinates
    if checker:
        checkers = []
        for sel in selected:
            #TODO replace the text beneath
            dest_min_row,dest_min_col= sel
            alt_row = (dest_min_row//skip)%2
            alt_col = (dest_min_col//skip)%2
                
            
            if alt_row ==0 and alt_col==1 or alt_row ==1 and alt_col ==0 :
                continue
            checkers.append(sel)
        selected = np.array(checkers)
    plot = plt.scatter(selected[:,1],selected[:,0])
    plt.imshow(part)
    for sel in selected:
        plt.gca().add_patch(plt.Rectangle((sel[1]-skip//2,sel[0]-skip//2),skip,skip,fill=False,color="red"))
    return [selected,coords]
    #plt.gca().invert_yaxis()

# selected,all_coords = get_zero_positions(part.numpy(),full,checker=True,skip=size,buffer=True)
# # modify to pair down to a small set of sections
# len(selected)

from tqdm import tqdm
# plt.figure()

# search for the parts of the image where the image is 

import json

def make_slice(lst):
    return slice(lst[0],lst[1]),slice(lst[2],lst[3])
def make_slice_numpy(arr):
    row,col = arr.tolist() # this is the order that d3 sends  it's extent

    return slice(*row),slice(*col)

def make_bbox_sized(sel,size):
    return [sel[0],sel[0]+size,sel[1],sel[1]+size]
# selected
def make_candidates(p,im,mask,zero_coords):
    im = pyvips.Image.new_from_array(im)
    row = p[0]
    col = p[1]
    # so we need 2 sets of coords, the inner and the inner bboxes
    #             outer
    # +-------------------+
    # |     col           |
    # |     -sz           |
    # |    +---------+    |
    # | row|         |    |
    # | -sz|  inner  |sz  |2sz
    # |    |         |    |
    # |    |         |    | xanchor
    # |    +---------xrow |
    # |        sz    col  |
    # |                   |
    # +-------------------x
    #         2sz
    # note that we need to start focusing on sampling "around" the location, not treating it as a corner like we do here
    min_inner_row = np.clip(row-half,0,im.height)
    max_inner_row = np.clip(row+half+1,0,im.height)
    min_inner_col = np.clip(col-half,0,im.width)
    max_inner_col = np.clip(col+half+1,0,im.width)
    min_outer_row = np.clip(row-half*4,0,im.height).astype("int")
    max_outer_row = np.clip(row + half*4 + 1,0,im.height).astype("int")
    min_outer_col = np.clip(col -half*4,0,im.width).astype("int")
    max_outer_col = np.clip(col + half*4 + 1,0,im.width).astype("int")
    #br stands for bottom right, note this is the coordinate we paste back into the image with the cloned section
    
    missing= [int(e) for e in [min_inner_row,max_inner_row,min_inner_col,max_inner_col]]
    
    # grab a larger section than we replace in order to make the spcor more sensitive to neighbors
    section = pyvips.Image.new_from_array(im.numpy()[min_outer_row:max_outer_row,min_outer_col:max_outer_col])
    # print("second spatial correlation part")
    spatial_corr = im.spcor(section)
    
    # print(section.width,section.height)
    # for some reason this single step takes most of the time of an iteration
    arr = spatial_corr.numpy()
    arr = arr.sum(axis=2)
    
    #making a kernel that we can attempt to add up the neighbors with
    
    selections = []
    # naturally remove the starting patch
    #print("place we want to fill is",col,row)
    # where we zero things out does depend on the "bottom right" overlay of section
    # coordinates that are the bottom right row and col inclusive
    # consider clipping to the dimensions of the image to prevent outside array indexing
    
    arr[min_outer_row:max_outer_row,min_outer_col:max_outer_col]=0
    # also throw out things that are within range of the edge because otherwise we get placed patches that aren't full tiles
    arr[:size,:] =0
    arr[-size:,:]=0
    arr[:,:size] =0
    arr[:,-size:]=0
    
    arr[zero_coords[:,0],zero_coords[:,1]]=0
    #arr[buffered_coords[:,0],buffered_coords[:,1]]=0
    arr = arr*mask
    # plt.figure()
    # plt.imshow(arr)
     # candidate collection
    max_cor = 0
    max_coord = []
        
    return {"missing":missing,"candidates":selections}
def multiprocessing_helper(all_args):
    p,im,mask,zero_coords = all_args
    return make_candidates(p,im,mask,zero_coords)
def calculate_patches(full,part,selected,zero_coords):
    
    #remove debug subset
    print(size)
    wonky = make_wonky(full)
    # think about whether we are using gray or not
    im = pyvips.Image.new_from_array(wonky)
    
    # ensure coords are within bounds of image
    # TODO handle corner case where we start at the top of the image, and can't sample anything really, because usually we anchor our sampling point as the bottom right
    # of teh patch
    # this will probably look like just skippin when either the row or col delta is zero
    # TODO consider  how the inner vs outer increase might affect our need to zero out larger portions of the spcor result (to prevent sampling at a black boundary)
    offset = [[r,c] for r in range(-double,double) for c in range(-double,double)]
    test = selected[:,None]
    # in theory this will create a larger buffer around the points where the zeros are for us to remove from the correlation results
    # should help to prevent selecting matching tiels partially on the black region
    # 
    
    # print("third zeroing stuff out")
    # TODO move this outside the loop because 
    
    buffered_coords = (test + offset).reshape(-1,2)
    # zero out the parts of the arr that we can't sample from in the part, so the areas that are 0
    buffered_coords[:,0] = np.clip(buffered_coords[:,0],0,full.height-1)
    buffered_coords[:,1] = np.clip(buffered_coords[:,1],0,full.width-1)
    #TODO double check that this is doing what we expect
    print(buffered_coords.shape)
    buffered_coords = np.unique(buffered_coords,axis=0)
    print(buffered_coords.shape)
    mask = np.ones((im.height,im.width))
    mask[buffered_coords[:,0],buffered_coords[:,1]]=0
    results = []
    for p in tqdm(selected):
        cands = make_candidates(p,im,mask,zero_coords)
        with open("buffered_checkered","a") as f:
            f.write(f"{cands}\n")
        results.append(cands)

    return results
    
    
# point_pairs = calculate_patches(full,part,selected,all_coords)

def select_num_candidates(arr,num,half):
    selections=[]
    for i in range(num):
        # perform separate candidate selecction for each of the channels of the arr
        
        coord = np.argmax(arr)
        
        #shape 0 is more like rows, 
        row,col = np.unravel_index(coord,arr.shape)
        value = arr[row,col]
        coords = [row-half,row+half+1,col-half,col+half+1]
        
        
        arr[row-half:row+half + 1,col-half:col+half+1] = 0
        # collect a source and dest arr sample
        
        #print("selected location for replacement",col,row)
        
        # these are the coordinates of the section's clone, ie the place somewhere else in the image that matches the full parts most like the missing area
        # importantly we really only want the inner part of the outer, so we should do some arithmetic here
        
        selections.append(coords)
        
        # fig = plt.figure()
        # plt.imshow(arr)
        # cir = plt.Circle((col,row),radius=5,color="red")
        # fig.axes[0].add_patch(cir)
        # hide this section so that we pick a new location next iteration
        # min row max row mincol max col
        # we want to zero out the image spcor which we matched the coordinate to
        #NOTE THIS EXPECTS THAT OUTER IS TWICE THE SIZE OF INNER, THIS MIGHT BE SOMETHING WE CHANGE
    return selections


# p = point_pairs[0]

## this section shows the regions of the cloning that have some b lack section sin them 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from PIL import Image 
import numpy as np 
half = size//2
# fig = plt.figure()

def fill_mod_copy(part,point_pairs,):
    mod_copy = part.copy()
    plt.imshow(part.numpy())
    for pp in point_pairs:
        dest_min_row,dest_max_row,dest_min_col, dest_max_col = pp["missing"]
        for cand in pp["candidates"]:
            try:
                start_row,end_row,start_col,end_col = cand # this is a bounding box string of pixel coords, min_row max row min col max col
                # print(dest_min_col,dest_min_row,start_col,start_row)
                
                # print(cand)
                # print(end_col -start_col,end_row-start_row)
                piece = part.crop(start_col,start_row,end_col -start_col,end_row-start_row)
                piece_arr = piece.numpy()[:,:,:3].sum(axis=2)
                #print(piece_arr)
                zeros = (piece_arr ==0).sum()
                if zeros >0:
                    # we have a piece with black in it
                    # rect = patches.Rectangle((start_col,start_row),size,size)
                    rect = patches.Rectangle((dest_min_col,dest_min_row), size, size, linewidth=1, 
                            edgecolor='r',fill=False) 
                    # #plt.gca.add_patch(rect)
                    axes = fig.get_axes()
                    axes[0].add_patch(rect)
                    mod_copy =mod_copy.insert(piece,dest_min_col,dest_min_row)
                    # maeka  blue box for the region where the sample was taken
                    rect = patches.Rectangle((start_col,start_row),end_col -start_col,end_row-start_row, linewidth=1, 
                            edgecolor='b',fill=False) 
                    # #plt.gca.add_patch(rect)
                    axes[0].add_patch(rect)

                    # plt.plot([dest_min_col,dest_min_row],[start_col,start_row],color="g",linewidth=1)
                    plt.plot([dest_min_col,start_col],[dest_min_row,start_row],color="g",linewidth=1)
                #mod_copy =mod_copy.insert(piece,dest_min_col,dest_min_row)
                break
            except:
                continue


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from PIL import Image 
import numpy as np 
half = size//2
# fig = plt.figure()

def debug_mod_copy_light(part,point_pairs):
    mod_copy = part.copy()
    plt.imshow(part.numpy())
    strange_cases = []
    for pp in point_pairs:
        dest_min_row,dest_max_row,dest_min_col, dest_max_col = pp["missing"]
        for cand in pp["candidates"]:
            try:
                start_row,end_row,start_col,end_col = cand # this is a bounding box string of pixel coords, min_row max row min col max col
                # print(dest_min_col,dest_min_row,start_col,start_row)
                
                # print(cand)
                # print(end_col -start_col,end_row-start_row)
                piece = part.crop(start_col,start_row,end_col -start_col,end_row-start_row)
                piece_arr = piece.numpy()[:,:,:3].sum(axis=2)
                #print(piece_arr)
                light = (piece_arr >550).sum()
                if light >0 :
                    strange_cases.append(pp)
                    # we have a piece with black in it
                    # rect = patches.Rectangle((setart_col,start_row),size,size)
                    rect = patches.Rectangle((dest_min_col,dest_min_row), size, size, linewidth=1, 
                            edgecolor='r',fill=False) 
                    # #plt.gca.add_patch(rect)
                    axes = fig.get_axes()
                    axes[0].add_patch(rect)
                    mod_copy =mod_copy.insert(piece,dest_min_col,dest_min_row)
                    # maeka  blue box for the region where the sample was taken
                    rect = patches.Rectangle((start_col,start_row),end_col -start_col,end_row-start_row, linewidth=1, 
                            edgecolor='b',fill=False) 
                    # #plt.gca.add_patch(rect)
                    axes[0].add_patch(rect)

                    # plt.plot([dest_min_col,dest_min_row],[start_col,start_row],color="g",linewidth=1)
                    plt.plot([dest_min_col,start_col],[dest_min_row,start_row],color="g",linewidth=1)
                #mod_copy =mod_copy.insert(piece,dest_min_col,dest_min_row)
                break
            except:
                continue

    # plt.imshow(mod_copy.numpy())