#!/bin/bash
cat models.tar.bz2.parta* > models.tar.bz2
tar -xf models.tar.bz2
rm models.tar.*
