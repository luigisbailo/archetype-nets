#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:16:09 2020

@author: oehlers
"""
a=b=1
c=True
print('hey{}'.format(tuple([a,b,c])))
def func(a,b,c): print(a,b,c)

lst = [1,2]
func('b',*lst)
a = b = 5
for a in range(2): print(a,b)