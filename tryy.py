#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:21:25 2020

@author: oehlers
"""

a = {('a','b'): "c"}
b = {}
print(a)
def newkey(newdict,collectdict):
    key, value = list(newdict.items())[0]
    key_len = len(key)
    
    if key in list(collectdict.keys()):
        key = key + (1,)
        while key in list(collectdict.keys()):
            key = key[:key_len] + (key[key_len] + 1,)
    
    collectdict.update({key: value})
    
for i in range(5):
    newkey(a,b) 
for key, value in b.items():
    print(key, value)