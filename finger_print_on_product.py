
def get_map_names_to_indices(list_samples):
    '''
    Calculate map to list of indices with same prefix filename
    '''
    list_samples = [x.split('__')[0] for x in list_samples]

    map_product_to_indeces={}
    for index,item in enumerate(list_samples):
        if item in map_product_to_indeces:        
            map_product_to_indeces[item].append(index)
        else:
            map_product_to_indeces[item] = [index]            
    return map_product_to_indeces


if __name__ == '__main__':
    list_samples = ["as__123","as__343","at__1233","at__sasd"]
    map_cal = get_map_names_to_indices(list_samples)
    print(map_cal)
