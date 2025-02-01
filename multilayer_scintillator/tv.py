import pylops
import autograd.numpy as np

def tv(shape, wrap=True):

    if not wrap:
        return pylops.Gradient(shape, kind='forward')
     
    L = np.prod(shape)
    
    oparr = []
    for i in range(len(shape)):

        grad = pylops.FirstDerivative(L, dims=shape, dir=i, kind='forward')
        
        get_edge = pylops.Restriction(L, [0, shape[i] - 1], dims=shape, dir=i)
        place_edge = pylops.Restriction(L, [shape[i] - 1, 0], dims=shape, dir=i).T
        edge_shape = shape[:i] + (2,) + shape[(i+1):]
        edge_grad = pylops.FirstDerivative(np.prod(edge_shape), dims=edge_shape, dir=i, kind='forward')
        
        oparr.append(grad + place_edge * edge_grad * get_edge)
    return pylops.VStack(oparr)

def get_connected_regions(dims, supp, wrap=True):
    regions = []
    all_explored = np.full(dims, False, dtype=bool)
    for start_pt in np.ndindex(dims):
        if all_explored[start_pt]:
            continue
        explored = set([start_pt])
        stack = [start_pt]
        while len(stack) > 0:
            pt = stack[-1]
            stack.pop()
            for i in range(len(dims)):
                if pt[i] < dims[i] - 1 or wrap:
                    new_pt = pt[:i] + ((pt[i]+1)%dims[i],) + pt[(i+1):]
                    if not supp[i][pt] and new_pt not in explored:
                        # print("going from {} to {}".format(pt, new_pt))
                        explored.add(new_pt)
                        stack.append(new_pt)
                if pt[i] > 0 or wrap:
                    new_pt = pt[:i] + ((pt[i]-1)%dims[i],) + pt[(i+1):]
                    if not supp[i][new_pt] and new_pt not in explored:
                        # print("going from {} to {}".format(pt, new_pt))
                        explored.add(new_pt)
                        stack.append(new_pt)  
        regions.append(list(explored))
        for pt in explored:
            all_explored[pt] = True
    return regions
    
class tv_proj(pylops.LinearOperator):
    def __init__(self, dims, supp):
        self.regions = get_connected_regions(dims, supp.reshape((len(dims), *dims)))
        self.shape = (len(self.regions), np.prod(dims))
        self.dtype = np.float_
        self.dims = dims
        self.explicit = False
    
    def _matvec(self, vec):
        vec = vec.reshape(self.dims)
        vec_proj = [sum(vec[pt] for pt in region) / np.sqrt(len(region)) for region in self.regions] # divide by sqrt to make orthonormal
        return np.array(vec_proj)
        
    def _rmatvec(self, vec_proj):
        vec = np.zeros(self.dims)
        for i, region in enumerate(self.regions):
            for pt in region:
                vec[pt] = vec_proj[i] / np.sqrt(len(region))
        return vec.flatten()

    def region_vals(self, vec):
        vec = vec.reshape(self.dims)
        return  np.array([sum(vec[pt] for pt in region) for region in self.regions])

def tv_proximal_approx(vec, dims, thresh):
    vec = vec.reshape(dims)
    D = len(dims)

    delta = np.zeros_like(vec)
    for i in range(D):
        vec_roll = np.roll(vec, -1, axis=i)
        diff = vec_roll - vec
        
        thresh_diff = np.minimum(2 * D * thresh, np.abs(diff) / 2) * np.sign(diff)
        delta += thresh_diff
        delta += -np.roll(thresh_diff, 1, axis=i)
        
    delta /= 2 * D

    return (vec + delta).flatten()
