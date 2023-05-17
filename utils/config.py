# coding: utf-8
import os
from collections import OrderedDict
CVC_300_root_test ='/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/CVC_300'
CVC_ClinicDB_root_test ='/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/CVC_ClinicDB'
CVC_ColonDB_root_test ='/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/CVC_ColonDB'
ETIS_LaribPolypDB_root_test ='/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/ETIS_LaribPolypDB'
Kvasir_root_test ='/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/Kvasir'

CVC_300 = os.path.join(CVC_300_root_test)
CVC_ClinicDB = os.path.join(CVC_ClinicDB_root_test)
CVC_ColonDB = os.path.join(CVC_ColonDB_root_test)
ETIS_LaribPolypDB = os.path.join(ETIS_LaribPolypDB_root_test)
Kvasir = os.path.join(Kvasir_root_test)

te_CVC_300 = os.path.join(CVC_300_root_test,"masks")
te_CVC_ClinicDB = os.path.join(CVC_ClinicDB_root_test,"masks")
te_CVC_ColonDB = os.path.join(CVC_ColonDB_root_test,"masks")
te_ETIS_LaribPolypDB = os.path.join(ETIS_LaribPolypDB_root_test,"masks")
te_Kvasir = os.path.join(Kvasir_root_test,"masks")

te_data_list = OrderedDict(
            {

                  "ColonDB": te_CVC_ColonDB,
                  "ETIS": te_ETIS_LaribPolypDB,
                  "Kvasir": te_Kvasir,
                  "CVC_300": te_CVC_300,
                  "ClinicDB": te_CVC_ClinicDB,

            },)

te_data_list_2 = OrderedDict(
            {   
                  "Kvasir": te_Kvasir,
                  "CVC_ClinicDB": te_CVC_ClinicDB,
                  "CVC_ColonDB": te_CVC_ColonDB,
                  "ETIS_LaribPolypDB": te_ETIS_LaribPolypDB,
                  "CVC_300": te_CVC_300,
      
            },)
# for name, path in te_data_list.items():
#     print(name)
#     print(path)
#     # print(te_data_list(name))
