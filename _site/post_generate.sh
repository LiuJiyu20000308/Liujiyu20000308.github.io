#!/bin/bash
dat=$(date +"%Y-%m-%d")
tim=$(date +"%Y-%m-%d %H:%M")
cd _posts
post=${dat}-${1}.md
touch $post
echo "---" >> $post 
echo "layout: post" >> $post 
echo "title: $1" >> $post 
echo "date: $tim +0800" >> $post 
echo "tags: [\"Learning Notes\", ]" >> $post 
echo "toc: true" >> $post 
echo "---" >> $post 
