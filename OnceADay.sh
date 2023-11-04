#!/bin/bash
rm -f ./_posts/*.html
echo "have removed all the html file in the './_posts' directory"
git add .
git commit -m "The day is over."
git push -u origin main
