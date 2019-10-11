import os

def load_in_images(file_path='ped1'):
    print_message = ""
    if file_path == 'ped1':
        path = './data/ped1/training/frames/'
        print_message = 'ped1'

    elif file_path == 'ped2':
        path = './data/ped2/training/frames/'
        print_message = 'ped2'
    else:
        path = './data/shanghaitech/training/frames/'
        print_message = 'shanghaitech'
    print("Loading in: {} Dataset".format(print_message))

    folders = sorted([os.path.join(path, f) for f in os.listdir(path)])

    overall_images = []
    for fol in folders:
        images = sorted([os.path.join(fol, img) for img in os.listdir(fol)])
        
        for idx, _ in enumerate(images):
            if idx < len(images) - 3:
                overall_images.append(images[idx: idx+4])
    
    return overall_images

