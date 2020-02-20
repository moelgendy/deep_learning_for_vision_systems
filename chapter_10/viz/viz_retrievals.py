import argparse
import csv
import os

import cv2
import h5py
import numpy as np

from annoy import AnnoyIndex


def parse_args():
    parser = argparse.ArgumentParser(description='top-k retrievals')
    parser.add_argument('--img_folder', dest='img_folder',
                        help='Path to image root containing image_query and image_test')
    parser.add_argument('--query_csv', dest='query_csv',
                        help='query csv file')
    parser.add_argument('--query_h5', dest='query_h5',
                        help='query .h5 file')
    parser.add_argument('--gallery_csv', dest='gallery_csv',
                        help='gallery csv file')
    parser.add_argument('--gallery_h5', dest='gallery_h5',
                        help='gallery .h5 file')
    parser.add_argument('--k', dest='top_k', type=int,
                        default=5)
    parser.add_argument('--output', dest='output',
                        help='Output path for images')
    return parser.parse_args()


def border_an_image(path_img, img_folder, value=[0, 255, 0]):
    q_img = cv2.imread(os.path.join(img_folder, path_img))
    q_img = cv2.resize(q_img, (128, 128))
    # border params - https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
    borderType = cv2.BORDER_CONSTANT
    top = int(0.05 * q_img.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * q_img.shape[1])  # shape[1] = cols
    right = left
    # make bordered image
    dst = cv2.copyMakeBorder(q_img, top, bottom, left,
                             right, borderType, None, value)
    return dst


def make_query_top_n_image(top_k_rets, path_q, label_q, paths_gallery, labels_gallery, img_folder):
    q_img = cv2.imread(os.path.join(img_folder, path_q))
    q_img = cv2.resize(q_img, (140, 140))
    output = q_img
    for opt in top_k_rets:
        _color = [0, 0, 255]
        if label_q == labels_gallery[opt]:
            _color = [0, 255, 0]
        g_img = border_an_image(paths_gallery[opt], img_folder, _color)
        g_img = cv2.resize(g_img, (140, 140))
        output = np.concatenate((output, g_img), axis=1)

    return output


def load_embedding_from_h5(path_query_h5, path_gallery_h5):
    with h5py.File(path_query_h5, 'r') as f_query:
        e_query = np.array(f_query['emb'])
    with h5py.File(path_gallery_h5, 'r') as f_gallery:
        e_gallery = np.array(f_gallery['emb'])
    return e_query, e_gallery


def load_path_labels_from_csv(csv_file):
    paths = []
    labels = []
    with open(csv_file, 'r') as fh:
        for row in csv.reader(fh, delimiter=','):
            paths.append(row[1])
            labels.append(row[0])
    return np.array(paths), np.array(labels)


def build_annoy_kd_tree(gallery_embs):
    t = AnnoyIndex(128, 'euclidean')
    count = 0
    for e in gallery_embs:
        t.add_item(count, e)
        count += 1
    t.build(256)
    return t


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # load image paths and labels (foldernames)
    paths_query, labels_query = load_path_labels_from_csv(args.query_csv)
    paths_gallery, labels_gallery = load_path_labels_from_csv(args.gallery_csv)

    # load embeddings
    embeddings_query, embeddings_gallery = load_embedding_from_h5(
        args.query_h5, args.gallery_h5)
    kd_tree = build_annoy_kd_tree(embeddings_gallery)

    for id_q, emb_q in enumerate(embeddings_query):
        k_rets = kd_tree.get_nns_by_vector(emb_q,
                                           args.top_k, search_k=-1, include_distances=False)
        output_img = make_query_top_n_image(
            k_rets, paths_query[id_q], labels_query[id_q], paths_gallery,
            labels_gallery, args.img_folder)
        output_img_path = os.path.join(args.output, str(id_q).zfill(6))
        output_img_path = output_img_path + ".png"
        #print("output is getting saved at: {}".format(output_img_path))
        assert(cv2.imwrite(output_img_path, output_img))
