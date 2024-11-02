import pdb
import src
import glob
import importlib
import os
import cv2



### Change path to images here
path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
###

all_submissions = glob.glob('./src/YashrajDeshmukh')
os.makedirs('./results/', exist_ok=True)
for idx,algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1],idx+1,len(all_submissions)))
    try:
        # module_name = '{}_{}'.format(algo.split(os.sep)[-1],'stitcher')
        module_name = '{}_{}'.format(os.path.basename(algo), 'stitcher')
        filepath = '{}{}stitcher.py'.format( algo,os.sep,'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        ###
        for impaths in glob.glob(path)[:1]:
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths)

            outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],spec.name)
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            cv2.imwrite(outfile,stitched_image)
            print(homography_matrix_list)
            print('Panaroma saved ... @ ./results/{}.png'.format(spec.name))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')
