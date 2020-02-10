# Variable split_file is obtained from downaloading the datasets
split_file = "/datasets/fashion/Evallist_eval_partition.txt"
# following are output path. please edit if needed
output_train_csv = "./in_shop_defense_triplet_loss_format_TRAIN.csv"
output_query_csv = "./in_shop_defense_triplet_loss_format_QUERY.csv"
output_gallery_csv = "./in_shop_defense_triplet_loss_format_GALLERY.csv"
id_mapper = {}
id_counter = -1

with open(split_file) as fp, open(output_train_csv, "w") as tr, open(output_gallery_csv, "w") as ga, open(output_query_csv, "w") as qu:
    line = fp.readline()
    cnt = 0
    while line:
        line = line.strip()  # remove leading and trailing whitespaces
        cnt += 1
        if cnt >= 3:
            metadata = []
            for tmp in line.split(" "):
                if len(tmp) is not 0:
                    metadata.append(tmp)
            print("metadata: {}".format(metadata))
            _path = metadata[0]
            _id = metadata[1]
            _categ = metadata[2]
            print("_categ: {}".format(_categ))
            assert(_categ == "train" or _categ ==
                   "query" or _categ == "gallery")
            if _id not in id_mapper.keys():
                id_counter += 1
                id_mapper[_id] = id_counter
            if _categ == "train":
                tmp_str = str(id_counter) + "," + _path
                tr.write(tmp_str)
                tr.write("\n")
            elif _categ == "query":
                tmp_str = str(id_counter) + "," + _path
                qu.write(tmp_str)
                qu.write("\n")
            elif _categ == "gallery":
                tmp_str = str(id_counter) + "," + _path
                ga.write(tmp_str)
                ga.write("\n")
            else:
                print("Not possible to reach here!! ")
        line = fp.readline()
