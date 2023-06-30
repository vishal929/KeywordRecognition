#!/bin/sh
# run our python script and periodically check log file and truncate on max size


python /home/vishal/Desktop/KeywordRecognition/PiSystem/main.py > /home/vishal/Desktop/KeywordLog.txt

while true
do
	SIZE = $(stat -c %s "/home/vishal/Desktop/KeywordLog.txt")
	if (SIZE -g $(20000 * 1024) )
	then
		# truncate the log file
		truncate -s 0 /home/vishal/Desktop/KeywordLog.txt
	fi	
	# sleeping so that this isnt a super busy while loop
	sleep(3600)
done
