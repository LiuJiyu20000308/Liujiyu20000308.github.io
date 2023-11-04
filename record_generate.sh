#!/bin/bash
dat=$(date +"%Y-%m-%d")
tim=$(date +"%Y-%m-%d %H:%M")
cd _posts
post=${dat}-Week_${1}_Record.md
touch $post
echo "---" >> $post 
echo "layout: post" >> $post 
echo "title: Week_$1 Record" >> $post 
echo "date: $tim +0800" >> $post 
echo "tags: [research_record]" >> $post 
echo "toc: true" >> $post 
echo "---" >> $post
echo "" >> $post
echo "My  week's record during postgraduate period." >> $post
echo "- Monday" >> $post
echo "" >> $post
echo "- Tuesday" >> $post
echo "" >> $post
echo "- Wednesday" >> $post
echo "" >> $post
echo "- Thursday" >> $post
echo "" >> $post
echo "- Friday" >> $post
echo "" >> $post
echo "- Saturday" >> $post
echo "" >> $post
echo "- Sunday" >> $post
echo "" >> $post
echo "- Summary" >> $post 
echo "" >> $post
echo "[comment]: <> ([See last week's record](https://zhengtongdu.github.io/2022/0//Week__Record/))" >> $post
echo "" >> $post
echo "[comment]: <> ([See next week's record](https://zhengtongdu.github.io/2022/0//Week__Record/))" >> $post
