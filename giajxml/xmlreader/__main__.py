'''
Created on 2021/10/20

@author: Sin Shimozono
'''

import json
import xmltodict
#from lxml.etree import parse

if __name__ == '__main__':
    # XML�t�@�C���̏���
    with open('../FG-GML-503032-AdmArea-20210701-0001.xml', encoding='utf-8') as fp:
        # xml�ǂݍ���
        xml_data = fp.read()
     
        # xml �� dict
        dict_data = xmltodict.parse(xml_data)
     
        print(dict_data)