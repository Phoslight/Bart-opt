#!/usr/bin/env bash

rm -rf pa.zip

find . -type f \
       -not -path "./.git/*" \
       -not -path "./out/*" \
       -not -path "*/.DS_Store" \
       -not -path "*/__pycache__/*" \
       -not -path "*/tmp/*" \
       -not -path "*/.idea/*" | zip pa.zip -@
