#!/bin/bash
rm -f ./_posts/*.html
git add .
git commit -m "$1"
git push -u origin master
