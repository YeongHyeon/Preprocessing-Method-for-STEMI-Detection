import os, inspect, glob, random, shutil
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self, setname, tr_ratio=0.9):

        self.data_path = os.path.join(PACK_PATH, "..", setname)
        self.tr_ratio = tr_ratio
        self.split_train_test()

        self.data4nn = os.path.join(PACK_PATH, "dataset")
        self.path_classes = glob.glob(os.path.join(self.data4nn, "train", "*"))
        self.path_classes.sort()
        self.class_names = []
        for path_class in self.path_classes: self.class_names.append(path_class.split("/")[-1])

        self.num_class = len(self.class_names)

        self.npy_train = glob.glob(os.path.join(self.data4nn, "train", "*", "*.npy"))
        self.npy_test = glob.glob(os.path.join(self.data4nn, "test", "*", "*.npy"))
        self.npy_train.sort()
        self.boundtxt = self.npy_train[0].split("/")[-2]
        for cidx, clsname in enumerate(self.class_names):
            if(clsname in self.boundtxt):
                bndidx = cidx
                break
        self.bounds = [0]
        for bi in range(bndidx):
            self.bounds.append(0)
        for idx, _ in enumerate(self.npy_train):
            if(self.boundtxt in self.npy_train[idx]): pass
            else:
                for cidx, clsname in enumerate(self.class_names):
                    if(clsname in self.npy_train[idx-1]):
                        idx_pri = cidx
                    if(clsname in self.npy_train[idx]):
                        idx_pos = cidx

                if(bndidx == idx_pri):
                    self.bounds.append(idx)
                    for _ in range(abs(idx_pri-idx_pos)-1):
                        self.bounds.append(idx)
                else:
                    for _ in range(abs(idx_pri-idx_pos)-1):
                        self.bounds.append(idx)
                    self.bounds.append(idx)
                self.boundtxt = self.npy_train[idx].split("/")[-2]
                bndidx = idx_pos
        for _ in range(abs(self.num_class-bndidx)-1):
            self.bounds.append(idx)

        random.shuffle(self.npy_train)
        self.npy_test.sort()

        self.idx_tr = 0
        self.idx_te = 0

        self.amount_tr = len(self.npy_train)
        self.amount_te = len(self.npy_test)
        print("Training: %d" %(self.amount_tr))
        print("Test: %d" %(self.amount_te))

        sample = np.load(self.npy_train[0])
        self.data_dim = sample.shape[0]
        self.channel = sample.shape[1]

    def makedir(self, path):
        try: os.mkdir(path)
        except: pass

    def split_train_test(self):

        subdirs = glob.glob(os.path.join(self.data_path, "*"))
        subdirs.sort()

        list_cls = []
        for sdix, sdir in enumerate(subdirs):
            npys = glob.glob(os.path.join(sdir, "*"))
            npys.sort()

            patnum = "_"
            patlist = []
            for nidx, npy in enumerate(npys):
                tmpnum = npy.split(",")[0].split("/")[-1]
                if(patnum == tmpnum): pass
                else:
                    patnum = tmpnum
                    patlist.append(patnum)

            numtr = int(len(patlist) * self.tr_ratio)

            random.shuffle(patlist)

            trtelist = []
            trtelist.append(patlist[:numtr])
            trtelist.append(patlist[numtr:])

            list_cls.append(trtelist)


        try: shutil.rmtree("dataset")
        except: pass
        self.makedir("dataset")
        self.makedir(os.path.join("dataset", "train"))
        self.makedir(os.path.join("dataset", "test"))

        for cidx, cdir in enumerate(subdirs):

            clsname = cdir.split("/")[-1]
            self.makedir(os.path.join("dataset", "train", clsname))
            self.makedir(os.path.join("dataset", "test", clsname))

            npylist = glob.glob(os.path.join(cdir, "*.npy"))
            npylist.sort()

            for fidx, foldlist in enumerate(list_cls[cidx]):
                for fold_content in foldlist:
                    for npyname in npylist:
                        if(fold_content in npyname):
                            tmp = np.load(npyname)
                            if(fidx == 1): np.save(os.path.join("dataset", "test", clsname, npyname.split("/")[-1]), tmp)
                            else: np.save(os.path.join("dataset", "train", clsname, npyname.split("/")[-1]), tmp)

    def split_cls(self, pathlist, bound_start, bound_end):

        try: return pathlist[bound_start:bound_end]
        except: return []

    def next_batch(self, batch_size=1, train=False):

        data = np.zeros((0, 1, 1))
        label = np.zeros((0, self.num_class))
        if(train):
            while(True):
                np_data = np.load(self.npy_train[self.idx_tr])
                for cidx, clsname in enumerate(self.class_names):
                    if(clsname in self.npy_train[self.idx_tr]): tmp_label = cidx
                label_vector = np.eye(self.num_class)[tmp_label]

                if(data.shape[0] == 0): data = np.zeros((0, np_data.shape[0], np_data.shape[1]))

                np_data = np.expand_dims(np_data, axis=0)
                label_vector = np.expand_dims(label_vector, axis=0)
                data = np.append(data, np_data, axis=0)
                label = np.append(label, label_vector, axis=0)

                if(data.shape[0] >= batch_size): break
                else: self.idx_tr = (self.idx_tr + 1) % self.amount_tr

            return data, label

        else:
            if(self.idx_te >= self.amount_te):
                self.idx_te = 0
                return None, None, None

            tmppath = self.npy_test[self.idx_te]
            np_data = np.load(self.npy_test[self.idx_te])
            for cidx, clsname in enumerate(self.class_names):
                if(clsname in self.npy_test[self.idx_te]): tmp_label = cidx
            try: label_vector = np.eye(self.num_class)[tmp_label]
            except: label_vector = np.zeros(self.num_class)

            if(data.shape[0] == 0): data = np.zeros((0, np_data.shape[0], np_data.shape[1]))

            np_data = np.expand_dims(np_data, axis=0)
            label_vector = np.expand_dims(label_vector, axis=0)
            data = np.append(data, np_data, axis=0)
            label = np.append(label, label_vector, axis=0)

            self.idx_te += 1

            return data, label, tmppath
