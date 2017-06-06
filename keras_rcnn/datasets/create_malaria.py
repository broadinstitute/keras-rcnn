import os.path
import pickle
import glob
import skimage.io
import xml.etree.ElementTree as ET


def load_data():
    """Creates pickle file of malaria data.
    # Returns
        Tuple of dictionaries: `train, validation, test`.
    """

    #extract data from xml file
    def extractObjectData(obj):
        deleted = int(obj.find('deleted').text)
        label = obj.find('name').text.strip()

        if deleted or (not label):
            raise Exception
        try:
            box = obj.find('segm').find('box')
            x = [int(box.find('xmin').text), int(box.find('xmax').text)]
            y = [int(box.find('ymin').text), int(box.find('ymax').text)]
        except:
            polygon = obj.find('polygon')
            x = []
            y = []
            for pt in polygon.findall('pt'):
                for px in pt.findall('x'):
                    x.extend([int(float(px.text))])
                for py in pt.findall('y'):
                    y.extend([int(float(py.text))])
        xmin, xmax = int(min(x)), int(max(x))
        ymin, ymax = int(min(y)), int(max(y))
        return xmin, ymin, xmax, ymax, label

    def get_data(xml):
        tree = ET.parse(xml)
        root = tree.getroot()
        data = []
        for obj in root.findall('object'):
            try:
                xmin, ymin, xmax, ymax, label = extractObjectData(obj)
                if label != 'rbc':
                    label = 'not'
                data.append({'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax, 'class': label})
            except:
                continue
        return data

    directory = '/data/research/object-detection/malaria/data/'
    dictionary = []

    for t in ['training', 'validation', 'test']:

        dir_ = os.path.join(directory, 'images', t)

        image_files = glob.glob(os.path.join(dir_,"*"))

        for image_file in image_files:
            x = {}
            x['filepath'] = os.path.basename(image_file)
            image = skimage.io.imread(image_file)
            x['width'] = image.shape[0]
            x['height'] = image.shape[1]
            basename, imagename = image_file.split('/images/')
            label_file = os.path.join(basename, 'labels', imagename.rsplit('.')[-1] + '.xml')
            x['bboxes'] = get_data(label_file)

        dictionary.append(x)

    return dictionary

data = load_data()
with open('malaria-data.pickle', 'wb') as f:
    pickle.dump(data, f)
print data